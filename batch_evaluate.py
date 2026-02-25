import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')

import time
import json
import csv
import numpy as np
import torch
import hydra
from termcolor import colored
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from common.parser import parse_cfg
from common.seed import set_seed
from CAG_eve.env import make_eval_batch_env
from tdmpc2 import TDMPC2

import gymnasium as gym


class LegacyStepAPIWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._last_info: Dict[str, Any] = {}

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        self._last_info = dict(info) if isinstance(info, dict) else {}
        return obs

    def step(self, action):
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
            info = dict(info) if isinstance(info, dict) else {}
            info.setdefault("terminated", bool(terminated))
            info.setdefault("truncated", bool(truncated))
            info.setdefault("done_for_learning", bool(terminated))
        elif isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            info = dict(info) if isinstance(info, dict) else {}
            info.setdefault("terminated", bool(done))
            info.setdefault("truncated", False)
            info.setdefault("done_for_learning", bool(done))
        else:
            raise RuntimeError(f"Unexpected step return: {out}")
        info.setdefault("success", bool(info.get("is_success", False)))
        info.setdefault("is_success", bool(info.get("success", False)))
        self._last_info = info
        return obs, float(reward), bool(done), info

    def rand_act(self):
        a = self.action_space.sample()
        return torch.as_tensor(a, dtype=torch.float32)


@dataclass
class EpisodeResult:
    target_idx: int
    target_point: List[float]
    episode_idx: int
    success: bool
    total_reward: float
    duration_seconds: float
    steps: int
    initial_distance: float             
    final_distance: float               
    distance_ratio: float               
    terminated: bool
    truncated: bool


@dataclass
class TargetSummary:
    target_idx: int
    target_point: List[float]
    n_episodes: int
    success_rate: float
    mean_reward: float
    mean_steps: float
    mean_duration: float
    mean_initial_distance: float
    mean_final_distance: float
    mean_distance_ratio: float          
    best_distance_ratio: float          
    episodes: List[EpisodeResult] = field(default_factory=list)


def extract_distance(info: Dict[str, Any], obs: Any, target: List[float]) -> float:
    for key in ("distance", "dist_to_goal", "final_distance", "dist"):
        if key in info:
            return float(info[key])
    if obs is not None and target is not None:
        try:
            obs_arr = np.array(obs).flatten()
            tgt_arr = np.array(target, dtype=np.float64)
            n = len(tgt_arr)
            if len(obs_arr) >= n:
                return float(np.linalg.norm(obs_arr[-n:] - tgt_arr))
        except Exception:
            pass
    return float("nan")


def safe_ratio(final: float, initial: float) -> float:
    if np.isnan(final) or np.isnan(initial) or initial == 0.0:
        return float("nan")
    return final / initial


def _unwrap_env(env):
    current = env
    while True:
        if hasattr(current, 'intervention'):
            return current
        if hasattr(current, 'env'):
            current = current.env
        else:
            raise AttributeError(
                f"Cannot find 'intervention' attribute in any layer of the env wrapper chain. "
                f"Innermost env type: {type(current)}"
            )


def evaluate_one_target(
    agent,
    env,
    target_cfg: Dict[str, Any],
    target_idx: int,
    n_episodes: int,
    max_steps: Optional[int],
    task_idx: Optional[int],
    device: torch.device,
    save_video: bool,
    video_dir: str,
) -> TargetSummary:
    target_point = target_cfg.get("target_point", [])
    target_label = target_cfg.get("label", f"target_{target_idx}")

    episodes: List[EpisodeResult] = []
    ep = 0  

    while ep < n_episodes:
        reset_kwargs = {}
        if target_point:
            reset_kwargs["options"] = {"target_point": target_point}
        obs = env.reset(**reset_kwargs)

        initial_distance = _unwrap_env(env).intervention.navigator.dist_to_target_scalar

        done = False
        ep_reward = 0.0
        steps = 0
        frames = []
        last_info: Dict[str, Any] = {}
        last_obs = obs
        t_start = time.perf_counter()
        simulation_error = False

        while not done:
            # print(f'ep0={ep}')
            action = agent.act(obs, t0=(steps == 0), task=task_idx)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            steps += 1
            if steps == 1:
                initial_distance = _unwrap_env(env).intervention.navigator.dist_to_target_scalar
            print(f"      step={steps:4d} | reward={reward:7.3f} | ep_reward={ep_reward:8.2f}", end='\r')
            last_info = info
            last_obs = obs

            if info.get("simulation_error", False):
                simulation_error = True

            if save_video and not simulation_error:  
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            if max_steps is not None and steps >= max_steps:
                break
                
        if simulation_error:
            print(colored(
                f"\n    [simulation_error] ep {ep:02d} aborted at step {steps}, retrying...",
                "magenta"
            ))
            continue 

        t_end = time.perf_counter()
        duration = t_end - t_start

        success = bool(last_info.get("success", last_info.get("is_success", False)))
        terminated = bool(last_info.get("terminated", done))
        truncated = bool(last_info.get("truncated", False))
        final_dist = extract_distance(last_info, last_obs, target_point)
        ratio = safe_ratio(final_dist, initial_distance)

        if save_video and frames:
            try:
                import imageio
                os.makedirs(video_dir, exist_ok=True)
                imageio.mimsave(os.path.join(video_dir, f"{target_label}_ep{ep}.mp4"), frames, fps=15)
            except Exception as e:
                print(f"  [warn] video save failed: {e}")

        res = EpisodeResult(
            target_idx=target_idx,
            target_point=list(target_point),
            episode_idx=ep,
            success=success,
            total_reward=ep_reward,
            duration_seconds=duration,
            steps=steps,
            initial_distance=initial_distance,
            final_distance=final_dist,
            distance_ratio=ratio,
            terminated=terminated,
            truncated=truncated,
        )
        episodes.append(res)

        status = colored("✓ SUCCESS", "green") if success else colored("✗ FAIL", "red")
        ratio_str = f"{ratio:.3f}" if not np.isnan(ratio) else " nan"
        print(
            f"    ep {ep:02d} | {status} | "
            f"steps={steps:4d} | reward={ep_reward:8.2f} | "
            f"init={initial_distance:.4f} → final={final_dist:.4f} "
            f"(ratio={ratio_str}) | time={duration:.2f}s"
        )

        ep += 1  
        
    success_rate        = float(np.mean([e.success for e in episodes]))
    mean_reward         = float(np.mean([e.total_reward for e in episodes]))
    mean_steps          = float(np.mean([e.steps for e in episodes]))
    mean_duration       = float(np.mean([e.duration_seconds for e in episodes]))

    valid_init  = [e.initial_distance for e in episodes if not np.isnan(e.initial_distance)]
    valid_final = [e.final_distance   for e in episodes if not np.isnan(e.final_distance)]
    valid_ratio = [e.distance_ratio   for e in episodes if not np.isnan(e.distance_ratio)]

    mean_initial_distance = float(np.mean(valid_init))  if valid_init  else float("nan")
    mean_final_distance   = float(np.mean(valid_final)) if valid_final else float("nan")
    mean_distance_ratio   = float(np.mean(valid_ratio)) if valid_ratio else float("nan")
    best_distance_ratio   = float(np.min(valid_ratio))  if valid_ratio else float("nan")

    return TargetSummary(
        target_idx=target_idx,
        target_point=list(target_point),
        n_episodes=n_episodes,
        success_rate=success_rate,
        mean_reward=mean_reward,
        mean_steps=mean_steps,
        mean_duration=mean_duration,
        mean_initial_distance=mean_initial_distance,
        mean_final_distance=mean_final_distance,
        mean_distance_ratio=mean_distance_ratio,
        best_distance_ratio=best_distance_ratio,
        episodes=episodes,
    )


def print_summary_table(summaries: List[TargetSummary]):
    sep = "─" * 110
    header = (
        f"{'Idx':>4}  {'Target Point':<28}  {'SuccRate':>8}  "
        f"{'MeanReward':>10}  {'MeanSteps':>9}  {'MeanTime':>9}  "
        f"{'InitDist':>9}  {'FinalDist':>9}  {'MeanRatio':>9}  {'BestRatio':>9}"
    )
    print(colored("\n" + sep, "cyan"))
    print(colored("  BATCH EVALUATION SUMMARY", "cyan", attrs=["bold"]))
    print(colored(sep, "cyan"))
    print(colored(header, "white", attrs=["bold"]))
    print(colored(sep, "cyan"))

    def _f(v, p=4):
        return "  N/A  " if np.isnan(v) else f"{v:.{p}f}"

    for s in summaries:
        tp_str = "[" + ", ".join(f"{v:.3f}" for v in s.target_point) + "]"
        succ_color = "green" if s.success_rate >= 0.5 else "red"
        ratio_color = "green" if (not np.isnan(s.mean_distance_ratio) and s.mean_distance_ratio < 0.5) else "yellow"
        print(
            f"{s.target_idx:>4}  {tp_str:<28}  "
            + colored(f"{s.success_rate*100:7.1f}%", succ_color) + "  "
            f"{s.mean_reward:>10.2f}  "
            f"{s.mean_steps:>9.1f}  "
            f"{s.mean_duration:>9.2f}  "
            f"{_f(s.mean_initial_distance):>9}  "
            f"{_f(s.mean_final_distance):>9}  "
            + colored(f"{_f(s.mean_distance_ratio, 3):>9}", ratio_color) + "  "
            + colored(f"{_f(s.best_distance_ratio, 3):>9}", "green")
        )
    print(colored(sep + "\n", "cyan"))


def save_csv(summaries: List[TargetSummary], path: str):
    rows = []
    for s in summaries:
        for e in s.episodes:
            def _fv(v):
                return round(v, 6) if not np.isnan(v) else "nan"
            rows.append({
                "target_idx":        e.target_idx,
                "target_point":      json.dumps(e.target_point),
                "episode_idx":       e.episode_idx,
                "success":           int(e.success),
                "total_reward":      round(e.total_reward, 4),
                "duration_seconds":  round(e.duration_seconds, 4),
                "steps":             e.steps,
                "initial_distance":  _fv(e.initial_distance),
                "final_distance":    _fv(e.final_distance),
                "distance_ratio":    _fv(e.distance_ratio),   # final / initial
                "terminated":        int(e.terminated),
                "truncated":         int(e.truncated),
            })
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(colored(f"  [CSV saved] {path}", "yellow"))


def save_json(summaries: List[TargetSummary], path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(s) for s in summaries], f, indent=2, default=str)
    print(colored(f"  [JSON saved] {path}", "yellow"))


def generate_html_report(summaries: List[TargetSummary], path: str):
    def _fmt(v, precision=4):
        if isinstance(v, float) and np.isnan(v):
            return "N/A"
        if isinstance(v, float):
            return f"{v:.{precision}f}"
        return str(v)

    def _ratio_bar(ratio):
        if isinstance(ratio, float) and np.isnan(ratio):
            return "<span style='color:#5a6480'>N/A</span>"
        pct = min(max(ratio * 100, 0), 100)
        r = int(min(pct * 2.55, 240))
        g = int(min((100 - pct) * 2.55, 200))
        color = f"rgb({r},{g},80)"
        return (
            f"<span style='display:inline-flex;align-items:center;gap:6px'>"
            f"<span style='display:inline-block;width:60px;height:7px;border-radius:4px;"
            f"background:var(--border);position:relative;overflow:hidden'>"
            f"<span style='position:absolute;left:0;top:0;height:100%;width:{pct:.1f}%;"
            f"background:{color};border-radius:4px'></span></span>"
            f"<span class='mono' style='font-size:0.78rem'>{ratio:.3f}</span>"
            f"</span>"
        )

    summary_rows = ""
    for s in summaries:
        tp_str = "[" + ", ".join(f"{v:.3f}" for v in s.target_point) + "]"
        sr_pct = s.success_rate * 100
        sr_class = "success" if sr_pct >= 50 else "fail"
        summary_rows += f"""
        <tr>
          <td class="mono">{s.target_idx}</td>
          <td class="mono small">{tp_str}</td>
          <td><span class="badge {sr_class}">{sr_pct:.1f}%</span></td>
          <td>{_fmt(s.mean_reward, 2)}</td>
          <td>{_fmt(s.mean_steps, 1)}</td>
          <td>{_fmt(s.mean_duration, 2)}s</td>
          <td class="mono">{_fmt(s.mean_initial_distance)}</td>
          <td class="mono">{_fmt(s.mean_final_distance)}</td>
          <td>{_ratio_bar(s.mean_distance_ratio)}</td>
          <td>{_ratio_bar(s.best_distance_ratio)}</td>
          <td>{s.n_episodes}</td>
        </tr>"""

    detail_rows = ""
    for s in summaries:
        tp_str = "[" + ", ".join(f"{v:.3f}" for v in s.target_point) + "]"
        for e in s.episodes:
            ok_class = "success" if e.success else "fail"
            detail_rows += f"""
        <tr>
          <td class="mono">{e.target_idx}</td>
          <td class="mono small">{tp_str}</td>
          <td>{e.episode_idx}</td>
          <td><span class="badge {ok_class}">{"✓" if e.success else "✗"}</span></td>
          <td>{_fmt(e.total_reward, 2)}</td>
          <td>{e.steps}</td>
          <td>{_fmt(e.duration_seconds, 2)}s</td>
          <td class="mono">{_fmt(e.initial_distance)}</td>
          <td class="mono">{_fmt(e.final_distance)}</td>
          <td>{_ratio_bar(e.distance_ratio)}</td>
          <td>{"term" if e.terminated else ("trunc" if e.truncated else "—")}</td>
        </tr>"""

    total_episodes  = sum(s.n_episodes for s in summaries)
    overall_success = np.mean([e.success for s in summaries for e in s.episodes]) * 100
    all_ratios      = [e.distance_ratio for s in summaries for e in s.episodes
                       if not np.isnan(e.distance_ratio)]
    overall_ratio   = np.mean(all_ratios) if all_ratios else float("nan")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Batch Evaluation Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
  :root {{
    --bg:#0d0f14; --surface:#13161d; --surface2:#1a1e28; --border:#252a38;
    --accent:#4af0b4; --text:#c8d0e0; --text-muted:#5a6480;
    --success:#3ddc84; --fail:#f06060;
  }}
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'IBM Plex Sans',sans-serif;background:var(--bg);color:var(--text);padding:0 0 60px}}
  .header{{background:linear-gradient(135deg,#0d1220,#0f1a2e);border-bottom:1px solid var(--border);padding:40px 48px 32px}}
  .header h1{{font-family:'IBM Plex Mono',monospace;font-size:1.7rem;color:var(--accent);letter-spacing:-.02em;margin-bottom:4px}}
  .header p{{color:var(--text-muted);font-size:.82rem;font-family:'IBM Plex Mono',monospace}}
  .stats-bar{{display:flex;gap:16px;padding:28px 48px;flex-wrap:wrap}}
  .stat-card{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px 24px;min-width:160px;flex:1}}
  .stat-card .label{{font-size:.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.1em;font-family:'IBM Plex Mono',monospace;margin-bottom:8px}}
  .stat-card .value{{font-family:'IBM Plex Mono',monospace;font-size:1.6rem;font-weight:600;color:var(--accent)}}
  .section{{padding:0 48px;margin-top:32px}}
  .section-title{{font-family:'IBM Plex Mono',monospace;font-size:.78rem;letter-spacing:.15em;text-transform:uppercase;color:var(--text-muted);margin-bottom:14px;display:flex;align-items:center;gap:10px}}
  .section-title::after{{content:'';flex:1;height:1px;background:var(--border)}}
  .table-wrap{{overflow-x:auto;border:1px solid var(--border);border-radius:10px}}
  table{{width:100%;border-collapse:collapse;font-size:.85rem}}
  thead th{{background:var(--surface2);padding:11px 16px;text-align:left;font-family:'IBM Plex Mono',monospace;font-size:.72rem;letter-spacing:.08em;color:var(--text-muted);border-bottom:1px solid var(--border);white-space:nowrap}}
  tbody td{{padding:10px 16px;border-bottom:1px solid var(--border);color:var(--text);vertical-align:middle}}
  tbody tr:last-child td{{border-bottom:none}}
  tbody tr:hover td{{background:var(--surface2)}}
  .mono{{font-family:'IBM Plex Mono',monospace}} .small{{font-size:.78rem}}
  .badge{{display:inline-block;padding:2px 9px;border-radius:4px;font-family:'IBM Plex Mono',monospace;font-size:.75rem;font-weight:600}}
  .badge.success{{background:rgba(61,220,132,.15);color:var(--success);border:1px solid rgba(61,220,132,.3)}}
  .badge.fail{{background:rgba(240,96,96,.12);color:var(--fail);border:1px solid rgba(240,96,96,.3)}}
  details summary{{cursor:pointer;user-select:none;padding:12px 0;color:var(--text-muted);font-family:'IBM Plex Mono',monospace;font-size:.78rem;letter-spacing:.1em;text-transform:uppercase;list-style:none;display:flex;align-items:center;gap:8px}}
  details summary::before{{content:'▶';font-size:.6rem;transition:transform .2s}}
  details[open] summary::before{{transform:rotate(90deg)}}
  details summary::after{{content:'';flex:1;height:1px;background:var(--border)}}
</style>
</head>
<body>
<div class="header">
  <h1>// BATCH EVALUATION REPORT</h1>
  <p>Targets: {len(summaries)} &nbsp;|&nbsp; Episodes: {total_episodes} &nbsp;|&nbsp; {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
<div class="stats-bar">
  <div class="stat-card"><div class="label">Targets</div><div class="value">{len(summaries)}</div></div>
  <div class="stat-card"><div class="label">Episodes</div><div class="value">{total_episodes}</div></div>
  <div class="stat-card"><div class="label">Overall Success</div>
    <div class="value" style="color:{'var(--success)' if overall_success>=50 else 'var(--fail)'}">{overall_success:.1f}%</div></div>
  <div class="stat-card"><div class="label">Mean Dist Ratio</div>
    <div class="value">{_fmt(overall_ratio, 3)}</div></div>
</div>
<div class="section">
  <div class="section-title">Per-Target Summary</div>
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>#</th><th>Target Point</th><th>Success Rate</th>
        <th>Mean Reward</th><th>Mean Steps</th><th>Mean Time</th>
        <th>Init Dist</th><th>Final Dist</th>
        <th>Mean Ratio (final/init)</th><th>Best Ratio</th><th>N</th>
      </tr></thead>
      <tbody>{summary_rows}</tbody>
    </table>
  </div>
</div>
<div class="section" style="margin-top:36px">
  <details>
    <summary>Episode Details ({total_episodes} rows)</summary>
    <div class="table-wrap" style="margin-top:12px">
      <table>
        <thead><tr>
          <th>Target#</th><th>Target Point</th><th>Ep</th><th>Result</th>
          <th>Reward</th><th>Steps</th><th>Time</th>
          <th>Init Dist</th><th>Final Dist</th>
          <th>Ratio (final/init)</th><th>End</th>
        </tr></thead>
        <tbody>{detail_rows}</tbody>
      </table>
    </div>
  </details>
</div>
</body>
</html>"""

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(colored(f"  [HTML saved] {path}", "yellow"))


def load_targets(targets_arg: str) -> List[Dict[str, Any]]:
    if os.path.isfile(targets_arg):
        with open(targets_arg) as f:
            data = json.load(f)
    else:
        data = json.loads(targets_arg)
    assert isinstance(data, list), "targets must be a JSON list"
    result = []
    for i, item in enumerate(data):
        if isinstance(item, (list, tuple)):
            result.append({"label": f"target_{i}", "target_point": list(item)})
        elif isinstance(item, dict):
            item.setdefault("label", f"target_{i}")
            result.append(item)
        else:
            raise ValueError(f"Unexpected target format at index {i}: {item}")
    return result


@hydra.main(config_name='CAG_config', config_path='.')
def batch_evaluate(cfg: dict):
    assert torch.cuda.is_available()
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    targets_arg = str(cfg.get("targets",    "targets.json"))
    n_episodes  = int(cfg.get("n_episodes", 3))
    max_steps   = cfg.get("max_steps",  None)
    output_dir  = str(cfg.get("output_dir", "./batch_eval_results"))
    save_video  = bool(cfg.get("save_video", False))

    if max_steps is not None:
        max_steps = int(max_steps)

    target_cfgs = load_targets(targets_arg)
    print(colored(
        f"\n[batch_evaluate] {len(target_cfgs)} target(s), "
        f"{n_episodes} ep each → {len(target_cfgs) * n_episodes} total episodes",
        "cyan", attrs=["bold"]
    ))
    print(colored(f"  checkpoint : {cfg.checkpoint}", "cyan"))
    print(colored(f"  output_dir : {output_dir}", "cyan"))

    env = make_eval_batch_env(cfg.env)
    env = LegacyStepAPIWrapper(env)

    if getattr(cfg, "action_dim", None) in (None, "???"):
        cfg.action_dim = int(np.prod(env.action_space.shape))
    if getattr(cfg, "obs_shape", None) in (None, "???"):
        ospec = env.observation_space
        if isinstance(ospec, gym.spaces.Box):
            cfg.obs_shape = tuple(ospec.shape)
        elif isinstance(ospec, gym.spaces.Dict):
            dim = sum(int(np.prod(sp.shape)) for sp in ospec.spaces.values()
                      if isinstance(sp, gym.spaces.Box))
            cfg.obs_shape = (dim,)
    if getattr(cfg, "episode_length", None) in (None, "???"):
        cfg.episode_length = (max_steps or
                              getattr(getattr(cfg, "env", None), "max_steps", None) or
                              200)
    if getattr(cfg, "multitask", None) in (None, "???"):
        cfg.multitask = False

    agent = TDMPC2(cfg)
    assert os.path.exists(cfg.checkpoint), f"Checkpoint not found: {cfg.checkpoint}"
    agent.load(cfg.checkpoint)
    print(colored(f"  checkpoint loaded.", "green"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_summaries: List[TargetSummary] = []
    for i, tcfg in enumerate(target_cfgs):
        label = tcfg.get("label", f"target_{i}")
        tp    = tcfg.get("target_point", [])
        print(colored(
            f"\n[{i+1}/{len(target_cfgs)}] {label}  target_point={tp}",
            "yellow", attrs=["bold"]
        ))
        summary = evaluate_one_target(
            agent=agent, env=env, target_cfg=tcfg,
            target_idx=i, n_episodes=n_episodes,
            max_steps=max_steps, task_idx=None,
            device=device, save_video=save_video,
            video_dir=os.path.join(output_dir, "videos", label),
        )
        all_summaries.append(summary)

    env.close()

    print_summary_table(all_summaries)
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    save_csv(all_summaries,             os.path.join(output_dir, f"results_{ts}.csv"))
    save_json(all_summaries,            os.path.join(output_dir, f"results_{ts}.json"))
    generate_html_report(all_summaries, os.path.join(output_dir, f"report_{ts}.html"))
    print(colored(f"\n  All outputs saved to: {os.path.abspath(output_dir)}\n", "cyan"))


if __name__ == '__main__':

    batch_evaluate()

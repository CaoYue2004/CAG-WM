import os
import threading
from typing import Callable, Optional

class FdTeeCapture:
    def __init__(self, fd: int, callback: Optional[Callable[[str], None]] = None):
        self.fd = fd
        self.callback = callback
        self._old_fd = None
        self._r = None
        self._w = None
        self._t = None
        self._stop = threading.Event()
        self._buf = b""

    def start(self):
        self._old_fd = os.dup(self.fd)
        self._r, self._w = os.pipe()
        os.dup2(self._w, self.fd)
        os.close(self._w)
        self._w = None

        def _reader():
            while not self._stop.is_set():
                try:
                    chunk = os.read(self._r, 4096)
                    if not chunk:
                        break
                except Exception:
                    break

                # 回显到原终端
                try:
                    os.write(self._old_fd, chunk)
                except Exception:
                    pass

                # 回调：尽量按行切，但即便没换行也会触发（用当前缓冲）
                if self.callback is not None:
                    self._buf += chunk
                    # 尝试按 \n 切行
                    while b"\n" in self._buf:
                        line, self._buf = self._buf.split(b"\n", 1)
                        try:
                            self.callback(line.decode(errors="replace"))
                        except Exception:
                            pass
                    # 另外：对“没换行”的情况，也给 callback 一份当前片段（可选）
                    # 你如果只想按行触发，就把下面这段注释掉
                    try:
                        self.callback(chunk.decode(errors="replace"))
                    except Exception:
                        pass

        self._t = threading.Thread(target=_reader, daemon=True)
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        try:
            if self._old_fd is not None:
                os.dup2(self._old_fd, self.fd)
                os.close(self._old_fd)
                self._old_fd = None
        except Exception:
            pass
        try:
            if self._r is not None:
                os.close(self._r)
                self._r = None
        except Exception:
            pass

# CAG-WM
Automated guidewire navigation in percutaneous coronary intervention can reduce radiation exposure and clinician workload. However, existing methods suffer from low sample efficiency and struggle with long-horizon guidewire navigation. World models offer a promising solution, but scarce paired angiographic-3D data leaves end-to-end world model approaches unexplored. To address these challenges, we propose CAG-WM, the first synthetic-data-driven world model for autonomous coronary guidewire navigation. Specifically, we construct dynamic synthetic coronary angiography–3D data and train a world model within a SOFA-based simulation environment to learn vessel–guidewire dynamics and perform internal multi-step rollouts for long-horizon navigation. CAG-WM is trained on 100 synthetic cases with full-view angiographic renderings and corresponding 3D voxelized coronary models, and evaluated on 20 multi-center clinical cases and 10 public datasets. Experimental results show that CAG-WM achieves a success rate exceeding 80\% in real coronary navigation, significantly outperforming competing methods. Even in unsuccessful cases, the guidewire remains close to the target lesion. Despite slightly longer per-step inference time, CAG-WM achieves the shortest total navigation time (about 2 min) thanks to its higher success rate, which reduces premature termination at the maximum step limit.



## Visualization of guidewire navigation by CAG-WM. 
![pre](https://github.com/user-attachments/assets/1d5685df-510b-4c15-903c-e12c68514d4b)
![pre_4](https://github.com/user-attachments/assets/787b42c3-8a04-480c-82cf-35320e9c53d9)
![pre_1](https://github.com/user-attachments/assets/56d058b3-b11d-4477-856c-c59e24e260de)

More experimental videos are provided in the supplementary materials.

## Failure cases of baseline methods.
![pre_3](https://github.com/user-attachments/assets/86ff87b6-a5f2-40c7-b4a9-c98d1f23c831)
![pre_4](https://github.com/user-attachments/assets/c0c8e8a0-1700-4c55-b5c8-b664223e9154)
![pre_2](https://github.com/user-attachments/assets/484dbc60-72c8-443d-92fc-58bebeaa9846)


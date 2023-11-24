# ENNMR

本项目在NMR的基础上，加入了系数矩阵的l-1范数，形成基于弹性网的核范数矩阵回归算法

## 项目介绍

main_AR.m: 使用AR数据集，测试遮挡性能。AR数据集AR_120_50_40.mat在项目中包含

main_yaleB.m:使用yaleB数据集，测试低光照性能。由于版权原因yaleB数据集不直接提供

ADMM_NMR.m:复现论文Yang J, Luo L, Qian J, et al. “Nuclear norm based matrix regression with applications to face recognition with occlusion and illumination changes”[J]. IEEE transactions on pattern analysis and machine intelligence, 2016, 39 (1): 156-171.的算法‘

ENNMR：项目的主体算法
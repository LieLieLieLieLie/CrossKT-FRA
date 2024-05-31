import warnings
warnings.filterwarnings("ignore")

from configs.args import *  # 选择配置（数据集，参数，gpu/cpu等等）
from utils import *  # 工具函数
from model.VFedPCA import *
from model.FedSVD import *
from model.AggVFL import *
from model.SplitVFL import *
from model.GAN import *
from performance import *

import time

if __name__ == '__main__':
    dir = r'model/save/Base/' + args.experiment2 + '/' + args.experiment3 + '/' + args.dataset + '/' + str(args.parameter) + '/CrossKTFRA/' + args.frl + '/'  # 保存路径
    os.makedirs(dir) if not os.path.exists(dir) else None
    print("实验保存路径：" + dir)

    device = get_gpu(args.gpu)  # 获取命令行的GPU设备
    experiment = Performance(device, dir)  # 任何需要训练模型的集中类
    # =========================================CrossKT-FDA实验前设置=========================================
    print('Dataset: ', args.dataset)
    print('Split method: ', args.split)
    # 仿真重叠数据都在X_task尾部
    X_task, y_task, X_shared, y_shared, X_data, y_data, X_task_shared, X_data_shared, num_classes = eval(args.split + '_split')(args.dataset)  # 仿真切分（数据）
    print('Task side: ', X_task.shape)
    print('Task side shared: ', X_task_shared.shape)
    print('Data side: ', X_data.shape)
    print('Data side shared: ', X_data_shared.shape)
    print('Shared samples', X_shared.shape)

    # start_time = time.time()  # 记录开始时间

    # Step 1: Federated Representation Learning
    frl_model_config = args.frl_params  # 调frl方法的配置参数
    frl_model = eval(args.frl)(num_features=X_shared.shape[1])  # 初始化frl方法类
    frl = FedRepresentationLearning(frl_model, frl_model_config, dir)  # 初始化综合数据学习类(frl方法类,frl方法配置参数)
    X_fed = frl.training(X_shared=X_shared, X_task_shared=X_task_shared, X_data_shared=X_data_shared)
    print("X_fed shape: ", X_fed.shape)
    # Step 2: Data side Train
    experiment.data_run(X_fed, y_shared, num_classes, args.data_type)
    print("联邦表示X_fed训练的模型:", experiment.data_acc)

    # Step 3: GAN generate
    task_extractor, task_classifier, _, _ = data_method_select(args.data_type, num_classes, X_fed.shape[1], X_fed.shape[1])  # 选择模型
    task_extractor.load_state_dict(torch.load(os.path.join(dir, 'data_extractor.pkl')))
    task_classifier.load_state_dict(torch.load(os.path.join(dir, 'data_classifier.pkl')))

    lrd_model_config = args.lrd_params  # 调lrd方法的配置参数
    lrd_model = GAN(X_task, X_fed, y_task, task_extractor, task_classifier, dir, device,  **lrd_model_config)
    lrd = LocalRepresentationDistillation(lrd_model, lrd_model_config, device, dir)
    lrd.train()  # y_task只用到了y_shared的部分，仿真方便CGAN的代码设计

    # end_time = time.time()  # 记录结束时间
    # execution_time = end_time - start_time  # 计算程序执行时间
    # print(f"CrossKT-FRA: {execution_time} 秒")
    # np.savez(f"{dir}/time.npz",
    #          time=execution_time, )

    X_task_new = lrd.representation_distillation_step(y_task)

    # Base对比
    # print("*******************Base对比*******************")
    # experiment.Base_run(X_task_shared, X_data_shared, y_shared, num_classes, X_fed, X_task_new)  # 两方共享数据，共享数据对应标签，类别数，联邦表示
    # experiment.Base_run(X_task_shared, X_data_shared, y_shared, num_classes, X_fed, X_task_new, y_task)  # 两方共享数据，共享数据对应标签，类别数，联邦表示

    # 消融设计
    print("*******************消融设计*******************")
    # experiment.Ablation_run_L(X_task_shared, X_data_shared, y_shared, num_classes, X_fed, X_task_new, y_task)  # 损失消融

    experiment.Ablation_run_private(X_task_shared, X_data_shared, y_shared, num_classes, X_fed, X_task_new, y_task, "Private")  # 有私有数据
    lrd_model_noPrivate = GAN(X_task, X_fed, y_task, task_extractor, task_classifier, dir, device,  **lrd_model_config)
    lrd_noPrivate = LocalRepresentationDistillation(lrd_model_noPrivate, lrd_model_config, device, dir)  # 私有数据消融
    lrd_noPrivate.train_noPrivate()  # y_task只用到了y_shared的部分，仿真方便CGAN的代码设计
    X_task_new_noPrivate = lrd_noPrivate.representation_distillation_step(y_task)
    experiment.Ablation_run_private(X_task_shared, X_data_shared, y_shared, num_classes, X_fed, X_task_new_noPrivate, y_task, "noPrivate")  # 无私有数据

    # 下游任务
    # print("*******************下游任务Local*******************")
    # experiment.Downstream_local_run(X_task, y_task, "Local", args.downstream)
    # experiment.Downstream_local_run(X_task_new, y_task, "CrossKTFRA", args.downstream)
    # print("*******************下游任务VFL*******************")
    # experiment.Downstream_VFL_run(X_task_shared, X_data_shared, y_shared, num_classes, X_fed, X_task_new)



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import json
from sklearn.utils import resample


# time new roman
plt.rcParams['font.family'] = 'Times New Roman'
class StatisticalAnalysis:

    def __init__(self):
        pass

    def rename_duplicate_indices(self, df):
        """
        重命名重复的index
        """
        new_index = []
        index_count = {}

        for idx in df.index:
            if idx in index_count:
                index_count[idx] += 1
                new_index.append(f"{idx}_{index_count[idx]}")
            else:
                index_count[idx] = 0
                new_index.append(idx)

        df.index = new_index
        return df

    def statistical_analysis(self, file_ai, file_human):
        data_ai = pd.read_excel(file_ai)
        data_human = pd.read_excel(file_human)

        # set 检查号为index
        data_ai.set_index('检查号', inplace=True)
        data_human.set_index('检查号', inplace=True)

        # # 如果有重复的检查号，则给第二个以后的加上'_1'、'_2'、'_3'...
        # data_ai = self.rename_duplicate_indices(data_ai)
        # data_human = self.rename_duplicate_indices(data_human)

        # 只取二者的交集
        data_ai = data_ai.loc[data_human.index]

        # 处理cad-rads分级
        data_ai['stage_cadras'].fillna('0', inplace=True)
        data_human['CAD-RADS'].fillna('0', inplace=True)
        data_ai['stage_cadras'] = data_ai['stage_cadras'].apply(lambda x: '0' if x == 'N' else x)
        data_ai['stage_cadras'].value_counts()
        data_human['CAD-RADS'].value_counts()

        # 处理p分级
        data_ai['stage_p'].value_counts()
        data_human['P分级'].value_counts()   
        data_ai['stage_p'].fillna('P1', inplace=True)
        # ai的p分级列，如果同一行的cad-rads分级为0，则p分级为0
        data_ai['stage_p'] = data_ai.apply(lambda x: 'P0' if x['stage_cadras'] == '0' else x['stage_p'], axis=1)
        data_human['P分级'].fillna(0, inplace=True)
        # 把data human中的p分级转换为字符串，1->P1, 2->P2, 3->P3 4->P4
        data_human['P分级'] = data_human['P分级'].apply(lambda x: f'P{int(x)}')
        data_human['P分级'].value_counts()

        # 处理冠脉开口
        data_ai['coronary_artery_abnormal'].value_counts()
        data_human['起源            （1=正常，0=异常）'].value_counts()
        # ai中的正常变成Normal，异常变成Abnormal
        data_ai['coronary_artery_abnormal'] = data_ai['coronary_artery_abnormal'].apply(lambda x: 'Normal' if x == "正常" else 'Abnormal')
        # hunam中的1变成Normal，0变成Abnormal
        data_human['起源            （1=正常，0=异常）'] = data_human['起源            （1=正常，0=异常）'].apply(lambda x: 'Normal' if x == 1 else 'Abnormal')

        # 处理心肌桥
        data_ai['myocardial_bridge'].value_counts()
        data_ai['myocardial_bridge'].fillna(0, inplace=True)
        data_human['心肌桥            （主要血管存在=1，不存在=0）'].value_counts()

        # 处理非钙化斑块
        data_ai['noncalcified_plaque'].value_counts()
        data_ai['noncalcified_plaque'].fillna(0, inplace=True)
        data_human['非钙化斑块         （存在=1，不存在=0）'].value_counts()

        # 将所有元素转换为字符串
        data_ai = data_ai.applymap(str)
        data_human = data_human.applymap(str)

        # 整合到一个dataframe，二者的index相同,要根据index合并
        data_ai.reset_index(inplace=True)
        data_human.reset_index(inplace=True)
        data = pd.concat([data_ai, data_human], axis=1, join='inner')

        # 混淆矩阵
        cm_cadrads = pd.crosstab(data['stage_cadras'], data['CAD-RADS'], rownames=['AI'], colnames=['Human'])
        cm_p = pd.crosstab(data['stage_p'], data['P分级'], rownames=['AI'], colnames=['Human'])
        cm_myocardial_bridge = pd.crosstab(data['myocardial_bridge'], data['心肌桥            （主要血管存在=1，不存在=0）'], rownames=['AI'], colnames=['Human'])
        cm_noncalcified_plaque = pd.crosstab(data['noncalcified_plaque'], data['非钙化斑块         （存在=1，不存在=0）'], rownames=['AI'], colnames=['Human'])
        # 创建大图和子图
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        self.plot_confusion_matrix(axs[0, 0], cm_cadrads, classes=cm_cadrads.columns, title='Confusion Matrix of CAD-RADS', cmap='Oranges')
        self.plot_confusion_matrix(axs[0, 1], cm_p, classes=cm_p.columns, title='Confusion Matrix of plaque burden', cmap='Oranges')
        self.plot_confusion_matrix(axs[1, 0], cm_myocardial_bridge, classes=cm_myocardial_bridge.columns, title='Confusion Matrix of myocardial bridge', cmap='Oranges')
        self.plot_confusion_matrix(axs[1, 1], cm_noncalcified_plaque, classes=cm_noncalcified_plaque.columns, title='Confusion Matrix of noncalcified plaque', cmap='Oranges')
        # 间距
        plt.subplots_adjust(hspace=0.5)
        # 创建大图和子图
        plt.tight_layout()
        plt.savefig('../data/Combined_Confusion_Matrices.tif', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
        plt.show()

        print(f"每个分级的数量（AI）：{data['stage_cadras'].value_counts()}")
        print(f"每个分级的数量（Human）：{data['CAD-RADS'].value_counts()}")

        # 找到人类评分cadrads为5，但是ai评分不为5的
        human5ainone5 = data[(data['CAD-RADS'] == '5') & (data['stage_cadras'] != '5')]
        # 找到人类是4b，但是ai评分不为4b的
        human4bainone4b = data[(data['CAD-RADS'] == '4b') & (data['stage_cadras'] != '4b')]
        # 找到人类是4a，但是ai评分不为4a的
        human4aainone4a = data[(data['CAD-RADS'] == '4a') & (data['stage_cadras'] != '4a')]
        # 找到人类是3，但是ai评分不为3的
        human3ainone3 = data[(data['CAD-RADS'] == '3') & (data['stage_cadras'] != '3')]
        # 找到人类是2，但是ai评分不为2的
        human2ainone2 = data[(data['CAD-RADS'] == '2') & (data['stage_cadras'] != '2')]
        # 找到人类是1，但是ai评分不为1的
        human1ainone1 = data[(data['CAD-RADS'] == '1') & (data['stage_cadras'] != '1')]
        # 找到人类是0，但是ai评分不为0的
        human0ainone0 = data[(data['CAD-RADS'] == '0') & (data['stage_cadras'] != '0')]
        # 保存
        human5ainone5.to_excel('../data/human5ainone5.xlsx')
        human4bainone4b.to_excel('../data/human4bainone4b.xlsx')
        human4aainone4a.to_excel('../data/human4aainone4a.xlsx')
        human3ainone3.to_excel('../data/human3ainone3.xlsx')
        human2ainone2.to_excel('../data/human2ainone2.xlsx')
        human1ainone1.to_excel('../data/human1ainone1.xlsx')
        human0ainone0.to_excel('../data/human0ainone0.xlsx')

        # 找到人类p分级为4，但是ai评分不为4的
        human4ainone4_p = data[(data['P分级'] == 'P4') & (data['stage_p'] != 'P4')]
        # 找到人类是3，但是ai评分不为3的
        human3ainone3_p = data[(data['P分级'] == 'P3') & (data['stage_p'] != 'P3')]
        # 找到人类是2，但是ai评分不为2的
        human2ainone2_p = data[(data['P分级'] == 'P2') & (data['stage_p'] != 'P2')]
        # 找到人类是1，但是ai评分不为1的
        human1ainone1_p = data[(data['P分级'] == 'P1') & (data['stage_p'] != 'P1')]
        # 找到人类是0，但是ai评分不为0的
        human0ainone0_p = data[(data['P分级'] == 'P0') & (data['stage_p'] != 'P0')]
        # 保存
        human4ainone4_p.to_excel('../data/human4ainone4_p.xlsx')
        human3ainone3_p.to_excel('../data/human3ainone3_p.xlsx')
        human2ainone2_p.to_excel('../data/human2ainone2_p.xlsx')
        human1ainone1_p.to_excel('../data/human1ainone1_p.xlsx')
        human0ainone0_p.to_excel('../data/human0ainone0_p.xlsx')

        # 计算每个CADRAS分级的accuracy、sensitivity、specificity、ppv、npv
        unique_ai_cadras = np.unique(data['stage_cadras'])
        unique_human_cadras = np.unique(data['CAD-RADS'])
        self.accuracy_cadrads = []
        self.sensitivity_cadrads = []
        self.specificity_cadrads = []
        self.f1score_cadrads = []

        for i in unique_ai_cadras:
            # data['stage_cadras'] 是预测值，data['CAD-RADS'] 是真实值
            tp = data[(data['stage_cadras'] == i) & (data['CAD-RADS'] == i)].shape[0]
            tn = data[(data['stage_cadras'] != i) & (data['CAD-RADS'] != i)].shape[0]
            fp = data[(data['stage_cadras'] == i) & (data['CAD-RADS'] != i)].shape[0]
            fn = data[(data['stage_cadras'] != i) & (data['CAD-RADS'] == i)].shape[0]

            # 计算accuracy
            self.accuracy_cadrads.append((tp + tn) / (tp + tn + fp + fn))
            # 计算sensitivity
            self.sensitivity_cadrads.append(tp / (tp + fn))
            # 计算specificity
            self.specificity_cadrads.append(tn / (tn + fp))
            # 计算f1 score
            self.f1score_cadrads.append(2 * tp / (2 * tp + fp + fn))

        print(f"accuracy_cadrads: {self.accuracy_cadrads}")
        print(f"sensitivity_cadrads: {self.sensitivity_cadrads}")
        print(f"specificity_cadrads: {self.specificity_cadrads}")
        print(f"f1score_cadrads: {self.f1score_cadrads}")

        # 计算每个P分级的accuracy、sensitivity、specificity、ppv、npv
        unique_ai_p = np.unique(data_ai['stage_p'])
        unique_human_p = np.unique(data_human['P分级'])
        self.accuracy_p = []
        self.sensitivity_p = []
        self.specificity_p = []
        self.f1score_p = []

        for i in unique_ai_p:
            # 计算accuracy
            tp = data[(data['stage_p'] == i) & (data['P分级'] == i)].shape[0]
            tn = data[(data['stage_p'] != i) & (data['P分级'] != i)].shape[0]
            fp = data[(data['stage_p'] == i) & (data['P分级'] != i)].shape[0]
            fn = data[(data['stage_p'] != i) & (data['P分级'] == i)].shape[0]

            # 计算accuracy
            self.accuracy_p.append((tp + tn) / (tp + tn + fp + fn))
            # 计算sensitivity
            self.sensitivity_p.append(tp / (tp + fn))
            # 计算specificity
            self.specificity_p.append(tn / (tn + fp))
            # 计算f1 score
            self.f1score_p.append(2 * tp / (2 * tp + fp + fn))

        print(f"accuracy_p: {self.accuracy_p}")
        print(f"sensitivity_p: {self.sensitivity_p}")
        print(f"specificity_p: {self.specificity_p}")
        print(f"f1score_p: {self.f1score_p}")

        # 计算心肌桥的accuracy、sensitivity、specificity、ppv、npv
        tp = data[(data['myocardial_bridge'] == '1.0') & (data['心肌桥            （主要血管存在=1，不存在=0）'] == '1')].shape[0]
        tn = data[(data['myocardial_bridge'] == '0.0') & (data['心肌桥            （主要血管存在=1，不存在=0）'] == '0')].shape[0]
        fp = data[(data['myocardial_bridge'] == '1.0') & (data['心肌桥            （主要血管存在=1，不存在=0）'] == '0')].shape[0]
        fn = data[(data['myocardial_bridge'] == '0.0') & (data['心肌桥            （主要血管存在=1，不存在=0）'] == '1')].shape[0]
        self.accuracy_myocardial_bridge = (tp + tn) / (tp + tn + fp + fn)
        self.sensitivity_myocardial_bridge = tp / (tp + fn)
        self.specificity_myocardial_bridge = tn / (tn + fp)
        self.f1score_myocardial_bridge = 2 * tp / (2 * tp + fp + fn)
        print(f"accuracy_myocardial_bridge: {self.accuracy_myocardial_bridge}")
        print(f"sensitivity_myocardial_bridge: {self.sensitivity_myocardial_bridge}")
        print(f"specificity_myocardial_bridge: {self.specificity_myocardial_bridge}")
        print(f"f1score_myocardial_bridge: {self.f1score_myocardial_bridge}")

        # 计算非钙化斑块的accuracy、sensitivity、specificity、ppv、npv
        tp = data[(data['noncalcified_plaque'] == '1.0') & (data['非钙化斑块         （存在=1，不存在=0）'] == '1')].shape[0]
        tn = data[(data['noncalcified_plaque'] == '0.0') & (data['非钙化斑块         （存在=1，不存在=0）'] == '0')].shape[0]
        fp = data[(data['noncalcified_plaque'] == '1.0') & (data['非钙化斑块         （存在=1，不存在=0）'] == '0')].shape[0]
        fn = data[(data['noncalcified_plaque'] == '0.0') & (data['非钙化斑块         （存在=1，不存在=0）'] == '1')].shape[0]
        self.accuracy_noncalcified_plaque = (tp + tn) / (tp + tn + fp + fn)
        self.sensitivity_noncalcified_plaque = tp / (tp + fn)
        self.specificity_noncalcified_plaque = tn / (tn + fp)
        self.f1score_noncalcified_plaque = 2 * tp / (2 * tp + fp + fn)

        print(f"accuracy_noncalcified_plaque: {self.accuracy_noncalcified_plaque}")
        print(f"sensitivity_noncalcified_plaque: {self.sensitivity_noncalcified_plaque}")
        print(f"specificity_noncalcified_plaque: {self.specificity_noncalcified_plaque}")
        print(f"f1score_noncalcified_plaque: {self.f1score_noncalcified_plaque}")

        # 保存所有的performance metrics到json文件
        performance_metrics = {
            'accuracy_cadrads': self.accuracy_cadrads,
            'sensitivity_cadrads': self.sensitivity_cadrads,
            'specificity_cadrads': self.specificity_cadrads,
            'f1score_cadrads': self.f1score_cadrads,
            'accuracy_p': self.accuracy_p,
            'sensitivity_p': self.sensitivity_p,
            'specificity_p': self.specificity_p,
            'f1score_p': self.f1score_p,
            'accuracy_myocardial_bridge': self.accuracy_myocardial_bridge,
            'sensitivity_myocardial_bridge': self.sensitivity_myocardial_bridge,
            'specificity_myocardial_bridge': self.specificity_myocardial_bridge,
            'f1score_myocardial_bridge': self.f1score_myocardial_bridge,
            'accuracy_noncalcified_plaque': self.accuracy_noncalcified_plaque,
            'sensitivity_noncalcified_plaque': self.sensitivity_noncalcified_plaque,
            'specificity_noncalcified_plaque': self.specificity_noncalcified_plaque,
            'f1score_noncalcified_plaque': self.f1score_noncalcified_plaque
        }
        with open('../data/performance_metrics.json', 'w') as f:
            json.dump(performance_metrics, f, indent=4)

        # Perform bootstrap analysis for 95% confidence intervals
        print("\n" + "="*60)
        print("Starting Bootstrap Analysis for 95% Confidence Intervals")
        print("="*60)
        self.confidence_intervals = self.bootstrap_analysis(data, n_bootstrap=1000)

        return self

    def plot_confusion_matrix(self, ax, cm, classes, title, cmap):
        """
        绘制混淆矩阵
        """
        sns.heatmap(cm, annot=True, 
                    fmt='d', 
                    linewidths=1,
                    annot_kws={"size": 12},
                    cmap=cmap, 
                    xticklabels=classes, 
                    yticklabels=classes,
                    cbar_kws={"shrink": 0.5, 'label': 'Number of samples'},
                    ax=ax)  # 使用传入的 ax

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)

        ax.set_xlabel(ax.get_xlabel(), fontsize=15)
        ax.set_ylabel(ax.get_ylabel(), fontsize=15)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
        ax.set_title(title, fontsize=20)

    def plot_performance_of_cadras(self, ax):
        categories = ['CAD-RADS 0', 'CAD-RADS 1', 'CAD-RADS 2', 'CAD-RADS 3', 'CAD-RADS 4A', 'CAD-RADS 4B', 'CAD-RADS 5']
        metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'F1 Score']
        data = [self.accuracy_cadrads, self.sensitivity_cadrads, self.specificity_cadrads, self.f1score_cadrads]

        x = np.arange(len(categories))
        width = 0.2

        for i, (metric, values) in enumerate(zip(metrics, data)):
            bars = ax.bar(x + i * width, values, width, label=metric, alpha=0.8)
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=12, rotation=0)

        ax.set_xticks(x + width * (len(metrics) - 1) / 2)
        # xticks 旋转
        ax.set_xticklabels(categories, rotation=0, ha='right')
        # ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1), ncol=1)
        ax.set_title('Performance Metrics Across Different CAD-RADS Categories', fontsize=20)
        ax.set_ylabel('Scores', fontsize=15)
        ax.set_xlabel('', fontsize=15)

        # ticklabel再大一点
        ax.tick_params(axis='x', labelsize=15)

    def plot_performance_of_p(self, ax):
        categories = ['P0', 'P1', 'P2', 'P3', 'P4']
        metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'F1 Score']
        data = [self.accuracy_p, self.sensitivity_p, self.specificity_p, self.f1score_p]

        x = np.arange(len(categories))
        width = 0.2
        for i, (metric, values) in enumerate(zip(metrics, data)):
            bars = ax.bar(x + i * width, values, width, label=metric, alpha=0.8)
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=12)

        ax.set_xticks(x + width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(categories)
        ax.legend(loc='upper center', bbox_to_anchor=(0.2, -0.05), ncol=4)
        ax.set_title('Performance Metrics Across Different P Categories', fontsize=20)
        ax.set_ylabel('', fontsize=15)
        ax.set_xlabel('', fontsize=15)

        # ticklabel再大一点
        ax.tick_params(axis='x', labelsize=15)

    def plot_cadrads_and_p(self):
        fig, axs = plt.subplots(2, 1, figsize=(15, 15))
        self.plot_performance_of_cadras(axs[0])
        self.plot_performance_of_p(axs[1])
        # 间距调整
        plt.subplots_adjust(hspace=0.8)
        plt.tight_layout()
        plt.savefig('../data/Combined_Performance_Metrics.tif', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
        plt.show()

    def plot_myocardial_bridge_and_noncalcified_plaque(self):
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        categories = ['Myocardial Bridge', 'Noncalcified Plaque']
        metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'F1 Score']
        data1 = [self.accuracy_myocardial_bridge, self.sensitivity_myocardial_bridge, self.specificity_myocardial_bridge, self.f1score_myocardial_bridge]
        data2 = [self.accuracy_noncalcified_plaque, self.sensitivity_noncalcified_plaque, self.specificity_noncalcified_plaque, self.f1score_noncalcified_plaque]

        width = 0.25
        axs[0].bar([0, 0.4, 0.8, 1.2], data1, width, label='Myocardial Bridge', alpha=0.8)
        axs[1].bar([0, 0.4, 0.8, 1.2], data2, width, label='Noncalcified Plaque', alpha=0.8)

        # xtickslaels
        axs[0].set_xticks([0, 0.4, 0.8, 1.2])
        axs[0].set_xticklabels(metrics)
        axs[0].set_title('Performance Metrics\nof Myocardial Bridge', fontsize=12)
        # xtickslaels
        axs[1].set_xticks([0, 0.4, 0.8, 1.2])
        axs[1].set_xticklabels(metrics)
        axs[1].set_title('Performance Metrics\nof Noncalcified Plaque', fontsize=12)

        # 添加数字
        for i, bar in enumerate(axs[0].patches):
            yval = bar.get_height()
            axs[0].text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)
        for i, bar in enumerate(axs[1].patches):
            yval = bar.get_height()
            axs[1].text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig('../data/Combined_Performance_Metrics_Myocardial_Bridge_Noncalcified_Plaque.tif', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})

    def bootstrap_analysis(self, data: pd.DataFrame, n_bootstrap: int = 1000) -> dict:
        """
        Perform bootstrap analysis to calculate 95% confidence intervals for all metrics
        
        Args:
            data: Combined dataframe with AI and human annotations
            n_bootstrap: Number of bootstrap samples (default: 1000)
            
        Returns:
            Dictionary containing 95% confidence intervals for all metrics
        """
        print(f"Performing bootstrap analysis with {n_bootstrap} samples...")
        
        # Get unique grades for CAD-RADS and P grading
        unique_cadrads = np.unique(data['stage_cadras'])
        unique_p = np.unique(data['stage_p'])
        
        # Initialize bootstrap results storage
        bootstrap_results = {
            'cadrads': {},
            'p_grading': {},
            'myocardial_bridge': {'accuracy': [], 'sensitivity': [], 'specificity': [], 'f1score': []},
            'noncalcified_plaque': {'accuracy': [], 'sensitivity': [], 'specificity': [], 'f1score': []}
        }
        
        # Initialize storage for each CAD-RADS grade
        for grade in unique_cadrads:
            bootstrap_results['cadrads'][f'grade_{grade}'] = {
                'accuracy': [], 'sensitivity': [], 'specificity': [], 'f1score': []
            }
        
        # Initialize storage for each P grade
        for grade in unique_p:
            bootstrap_results['p_grading'][f'grade_{grade}'] = {
                'accuracy': [], 'sensitivity': [], 'specificity': [], 'f1score': []
            }
        
        # Progress tracking
        for i in range(n_bootstrap):
            if (i + 1) % 100 == 0:
                print(f"Bootstrap iteration: {i + 1}/{n_bootstrap}")
            
            # Bootstrap resample with replacement
            bootstrap_data = resample(data, random_state=i)
            
            # Calculate metrics for each CAD-RADS grade individually
            for grade in unique_cadrads:
                tp = bootstrap_data[(bootstrap_data['stage_cadras'] == grade) & (bootstrap_data['CAD-RADS'] == grade)].shape[0]
                tn = bootstrap_data[(bootstrap_data['stage_cadras'] != grade) & (bootstrap_data['CAD-RADS'] != grade)].shape[0]
                fp = bootstrap_data[(bootstrap_data['stage_cadras'] == grade) & (bootstrap_data['CAD-RADS'] != grade)].shape[0]
                fn = bootstrap_data[(bootstrap_data['stage_cadras'] != grade) & (bootstrap_data['CAD-RADS'] == grade)].shape[0]
                
                # Calculate metrics for this specific grade
                accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                f1score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
                
                # Store results for this grade
                bootstrap_results['cadrads'][f'grade_{grade}']['accuracy'].append(accuracy)
                bootstrap_results['cadrads'][f'grade_{grade}']['sensitivity'].append(sensitivity)
                bootstrap_results['cadrads'][f'grade_{grade}']['specificity'].append(specificity)
                bootstrap_results['cadrads'][f'grade_{grade}']['f1score'].append(f1score)
            
            # Calculate metrics for each P grade individually
            for grade in unique_p:
                tp = bootstrap_data[(bootstrap_data['stage_p'] == grade) & (bootstrap_data['P分级'] == grade)].shape[0]
                tn = bootstrap_data[(bootstrap_data['stage_p'] != grade) & (bootstrap_data['P分级'] != grade)].shape[0]
                fp = bootstrap_data[(bootstrap_data['stage_p'] == grade) & (bootstrap_data['P分级'] != grade)].shape[0]
                fn = bootstrap_data[(bootstrap_data['stage_p'] != grade) & (bootstrap_data['P分级'] == grade)].shape[0]
                
                # Calculate metrics for this specific grade
                accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                f1score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
                
                # Store results for this grade
                bootstrap_results['p_grading'][f'grade_{grade}']['accuracy'].append(accuracy)
                bootstrap_results['p_grading'][f'grade_{grade}']['sensitivity'].append(sensitivity)
                bootstrap_results['p_grading'][f'grade_{grade}']['specificity'].append(specificity)
                bootstrap_results['p_grading'][f'grade_{grade}']['f1score'].append(f1score)
            
            # Calculate metrics for myocardial bridge
            tp_mb = bootstrap_data[(bootstrap_data['myocardial_bridge'] == '1.0') & (bootstrap_data['心肌桥            （主要血管存在=1，不存在=0）'] == '1')].shape[0]
            tn_mb = bootstrap_data[(bootstrap_data['myocardial_bridge'] == '0.0') & (bootstrap_data['心肌桥            （主要血管存在=1，不存在=0）'] == '0')].shape[0]
            fp_mb = bootstrap_data[(bootstrap_data['myocardial_bridge'] == '1.0') & (bootstrap_data['心肌桥            （主要血管存在=1，不存在=0）'] == '0')].shape[0]
            fn_mb = bootstrap_data[(bootstrap_data['myocardial_bridge'] == '0.0') & (bootstrap_data['心肌桥            （主要血管存在=1，不存在=0）'] == '1')].shape[0]
            
            accuracy_mb = (tp_mb + tn_mb) / (tp_mb + tn_mb + fp_mb + fn_mb) if (tp_mb + tn_mb + fp_mb + fn_mb) > 0 else 0
            sensitivity_mb = tp_mb / (tp_mb + fn_mb) if (tp_mb + fn_mb) > 0 else 0
            specificity_mb = tn_mb / (tn_mb + fp_mb) if (tn_mb + fp_mb) > 0 else 0
            f1score_mb = 2 * tp_mb / (2 * tp_mb + fp_mb + fn_mb) if (2 * tp_mb + fp_mb + fn_mb) > 0 else 0
            
            bootstrap_results['myocardial_bridge']['accuracy'].append(accuracy_mb)
            bootstrap_results['myocardial_bridge']['sensitivity'].append(sensitivity_mb)
            bootstrap_results['myocardial_bridge']['specificity'].append(specificity_mb)
            bootstrap_results['myocardial_bridge']['f1score'].append(f1score_mb)
            
            # Calculate metrics for noncalcified plaque
            tp_ncp = bootstrap_data[(bootstrap_data['noncalcified_plaque'] == '1.0') & (bootstrap_data['非钙化斑块         （存在=1，不存在=0）'] == '1')].shape[0]
            tn_ncp = bootstrap_data[(bootstrap_data['noncalcified_plaque'] == '0.0') & (bootstrap_data['非钙化斑块         （存在=1，不存在=0）'] == '0')].shape[0]
            fp_ncp = bootstrap_data[(bootstrap_data['noncalcified_plaque'] == '1.0') & (bootstrap_data['非钙化斑块         （存在=1，不存在=0）'] == '0')].shape[0]
            fn_ncp = bootstrap_data[(bootstrap_data['noncalcified_plaque'] == '0.0') & (bootstrap_data['非钙化斑块         （存在=1，不存在=0）'] == '1')].shape[0]
            
            accuracy_ncp = (tp_ncp + tn_ncp) / (tp_ncp + tn_ncp + fp_ncp + fn_ncp) if (tp_ncp + tn_ncp + fp_ncp + fn_ncp) > 0 else 0
            sensitivity_ncp = tp_ncp / (tp_ncp + fn_ncp) if (tp_ncp + fn_ncp) > 0 else 0
            specificity_ncp = tn_ncp / (tn_ncp + fp_ncp) if (tn_ncp + fp_ncp) > 0 else 0
            f1score_ncp = 2 * tp_ncp / (2 * tp_ncp + fp_ncp + fn_ncp) if (2 * tp_ncp + fp_ncp + fn_ncp) > 0 else 0
            
            bootstrap_results['noncalcified_plaque']['accuracy'].append(accuracy_ncp)
            bootstrap_results['noncalcified_plaque']['sensitivity'].append(sensitivity_ncp)
            bootstrap_results['noncalcified_plaque']['specificity'].append(specificity_ncp)
            bootstrap_results['noncalcified_plaque']['f1score'].append(f1score_ncp)
        
        # Calculate 95% confidence intervals
        confidence_intervals = {}
        
        # Process CAD-RADS grades
        confidence_intervals['cadrads'] = {}
        for grade_key in bootstrap_results['cadrads']:
            confidence_intervals['cadrads'][grade_key] = {}
            for metric in bootstrap_results['cadrads'][grade_key]:
                values = bootstrap_results['cadrads'][grade_key][metric]
                ci_lower = np.percentile(values, 2.5)
                ci_upper = np.percentile(values, 97.5)
                confidence_intervals['cadrads'][grade_key][metric] = {
                    'lower': ci_lower,
                    'upper': ci_upper,
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        
        # Process P grades
        confidence_intervals['p_grading'] = {}
        for grade_key in bootstrap_results['p_grading']:
            confidence_intervals['p_grading'][grade_key] = {}
            for metric in bootstrap_results['p_grading'][grade_key]:
                values = bootstrap_results['p_grading'][grade_key][metric]
                ci_lower = np.percentile(values, 2.5)
                ci_upper = np.percentile(values, 97.5)
                confidence_intervals['p_grading'][grade_key][metric] = {
                    'lower': ci_lower,
                    'upper': ci_upper,
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        
        # Process myocardial bridge and noncalcified plaque
        for category in ['myocardial_bridge', 'noncalcified_plaque']:
            confidence_intervals[category] = {}
            for metric in bootstrap_results[category]:
                values = bootstrap_results[category][metric]
                ci_lower = np.percentile(values, 2.5)
                ci_upper = np.percentile(values, 97.5)
                confidence_intervals[category][metric] = {
                    'lower': ci_lower,
                    'upper': ci_upper,
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        
        # Save confidence intervals to JSON file
        with open('../data/confidence_intervals.json', 'w') as f:
            json.dump(confidence_intervals, f, indent=4)
        
        # Print results
        print("\n95% Confidence Intervals:")
        print("=" * 70)
        
        # Print CAD-RADS results by grade
        print("\nCAD-RADS GRADING:")
        for grade_key in sorted(confidence_intervals['cadrads'].keys()):
            grade_name = grade_key.replace('grade_', 'CAD-RADS ')
            print(f"\n  {grade_name}:")
            for metric in ['accuracy', 'sensitivity', 'specificity', 'f1score']:
                ci = confidence_intervals['cadrads'][grade_key][metric]
                print(f"    {metric.capitalize()}: {ci['mean']:.3f} (95% CI: {ci['lower']:.3f}-{ci['upper']:.3f})")
        
        # Print P grading results by grade
        print("\nP GRADING:")
        for grade_key in sorted(confidence_intervals['p_grading'].keys()):
            grade_name = grade_key.replace('grade_', '')
            print(f"\n  {grade_name}:")
            for metric in ['accuracy', 'sensitivity', 'specificity', 'f1score']:
                ci = confidence_intervals['p_grading'][grade_key][metric]
                print(f"    {metric.capitalize()}: {ci['mean']:.3f} (95% CI: {ci['lower']:.3f}-{ci['upper']:.3f})")
        
        # Print other results
        for category in ['myocardial_bridge', 'noncalcified_plaque']:
            category_name = category.replace('_', ' ').title()
            print(f"\n{category_name.upper()}:")
            for metric in ['accuracy', 'sensitivity', 'specificity', 'f1score']:
                ci = confidence_intervals[category][metric]
                print(f"  {metric.capitalize()}: {ci['mean']:.3f} (95% CI: {ci['lower']:.3f}-{ci['upper']:.3f})")
        
        return confidence_intervals

if __name__ == '__main__':
    file_ai = your_file_ai
    file_human = your_file_human
    ss = StatisticalAnalysis()
    
    # Perform statistical analysis including bootstrap confidence intervals
    ss.statistical_analysis(file_ai, file_human)
    
    # Generate performance plots
    ss.plot_cadrads_and_p()
    ss.plot_myocardial_bridge_and_noncalcified_plaque()
    
    print("\nAnalysis completed! Results saved to:")
    print("- Performance metrics: ../data/performance_metrics.json")
    print("- 95% Confidence intervals: ../data/confidence_intervals.json")
    print("- Confusion matrices plot: ../data/Combined_Confusion_Matrices.tif")
    print("- Performance metrics plots: ../data/Combined_Performance_Metrics.tif")
    print("- Additional metrics plot: ../data/Combined_Performance_Metrics_Myocardial_Bridge_Noncalcified_Plaque.tif")
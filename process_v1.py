"""
从AI的response中提取出需要的信息
"""

import pandas as pd
import numpy as np
import os
import json
import re


class Process:
    def __init__(self):
        pass
    
    def read_data(self, file, sheet_name='GPT4结果'):
        """
        读取数据
        :param file: 数据文件
        :return: 数据
        """
        if os.path.exists(file) is False:
            raise FileNotFoundError(f'文件不存在：{file}')
        
        data = pd.read_excel(file, sheet_name=sheet_name, engine='openpyxl')
        return data

    def extract_key_value(self, text, idx):
        if 2485838 == idx:
            print(text)
        target_keys = ["分段编号", "名称", "狭窄程度", "斑块类型", "是否存在心肌桥", "修饰符"]
        
        # 假设你的JSON文本存储在text变量中
        pattern = r'.*:\s*(\[.*?\])\s*}'  # 匹配 "segments": [ ... ]
        # 使用re.DOTALL标志使.能匹配换行符
        match = re.search(pattern, text, re.DOTALL)
        if match:
            segments_list = match.group(1)  # 获取匹配的列表部分

        # 先根据{}将segments_list分割成多个段落
        segments_list = segments_list.split('},')

        # results = []
        # for iseg, seg in enumerate(segments_list):
        #     result = {}
        #     for key in target_keys:
        #         try:
        #             # 查找每个key的值
        #             if f'"{key}": "' in seg:  # 字符串类型的值
        #                 value = seg.split(f'"{key}": "')[1].split('"')[0]
        #             elif f'"{key}": ' in seg:  # 数字类型的值
        #                 value = seg.split(f'"{key}": ')[1].split(',')[0]
        #             result.update({key: value})
        #         except:
        #             result.update({key: np.nan})
            
        #     results.append(result)

        # 正则表达式提取键值对
        key_value_pattern = r'"(.*?)":\s*(".*?"|\d+)'
        results = []
        for iseg, seg in enumerate(segments_list):
            result = {}
            for key in target_keys:
                try:
                    matches = re.findall(key_value_pattern, seg)
                    for match in matches:
                        result.update({match[0]: match[1]})
                except:
                    result.update({key: np.nan})
            results.append(result)

        return results

    def extract(self, response):
        """
        从AI的response中提取出需要的信息
        :param response: AI的response
        :return: 提取出的信息
        """
        # index设置为检查号
        response = response.set_index('检查号')

        # 去掉重复的index
        response = response[~response.index.duplicated(keep='first')]

        # 提取冠脉优势：【"冠状动脉优势型": "右冠优势型"】则提取出右冠优势型
        coronary_artery_dominance = response['AI输出'].apply(lambda x: re.findall(r'冠状动脉优势型": "(.*?)"', x))
        coronary_artery_dominance = coronary_artery_dominance.apply(lambda x: x[0] if len(x) > 0 else np.nan)

        # 提取冠脉异常与否："冠状动脉开口": "正常"则提取出正常
        coronary_artery_abnormal = response['AI输出'].apply(lambda x: re.findall(r'冠状动脉开口": "(.*?)"', x))
        coronary_artery_abnormal = coronary_artery_abnormal.apply(lambda x: x[0] if len(x) > 0 else '')
        # 只要包含正常两个字，就是正常
        coronary_artery_abnormal = coronary_artery_abnormal.apply(lambda x: '正常' if '正常' in x else x)
        # 空值填充为正常
        coronary_artery_abnormal = coronary_artery_abnormal.replace('', '正常')
        # print多少不正常
        print(coronary_artery_abnormal.value_counts())

        segments = {}
        for idx in response.index:
            text = response.loc[idx, 'AI输出']
            try:
                dict_ = self.extract_key_value(text, idx)
                segments[idx] = dict_
            except Exception as e:
                print(f'{idx} has Error: {e}')
                continue
        
        # 把狭窄程度转换为数字
        value_pattern = r'\d+(?:\.\d+)?'
        for idx in segments.keys():
            if 2485838 == idx:
                print(segments[idx])
            for i in range(min(len(segments[idx]), 18)):
                
                # # 如果字符串包含非数字，则打印
                # if not segments[idx][i]['狭窄程度'].isdigit():
                #     print(idx, segments[idx][i]['狭窄程度'])

                # '0.4-0.5' -> 0.5，取最高值
                if isinstance(segments[idx][i]['狭窄程度'], str):
                    # 如果有>，取>后面的数字
                    if '>' in segments[idx][i]['狭窄程度']:
                        segments[idx][i]['狭窄程度'] = float(re.findall(value_pattern, segments[idx][i]['狭窄程度'])[0])
                    elif '闭塞' in segments[idx][i]['狭窄程度']:
                        segments[idx][i]['狭窄程度'] = 1
                    elif '轻微' in segments[idx][i]['狭窄程度']:
                        segments[idx][i]['狭窄程度'] = 0.1
                    elif '无' in segments[idx][i]['狭窄程度']:
                        segments[idx][i]['狭窄程度'] = 0
                    elif '-' in segments[idx][i]['狭窄程度']:
                        pattern = r'\d+\.\d+-(\d+\.\d+)'
                        match = re.search(pattern, segments[idx][i]['狭窄程度'])
                        second_number = match.group(1)
                        segments[idx][i]['狭窄程度'] = float(second_number)

                    # 其它中文则狭窄程度为0
                    elif re.search(r'[\u4e00-\u9fa5]', segments[idx][i]['狭窄程度']):
                        segments[idx][i]['狭窄程度'] = 0
                    else:
                        segments[idx][i]['狭窄程度'] = float(re.findall(value_pattern, segments[idx][i]['狭窄程度'])[0])

        # save to json
        with open('../data/segments.json', 'w', encoding='utf-8') as f:
            json.dump(segments, f, indent=4, ensure_ascii=False)

        return segments, coronary_artery_dominance, coronary_artery_abnormal

    def get_stage(self, segments):
        """
        从AI的response中提取出分级信息
        :param segments: AI的response(预处理后的)
        :return: 分级信息
        """
        value_pattern = r'\d+(?:\.\d+)?'

        # CADRADS分级
        ## 提取分段编号和狭窄程度
        result_stenosis = {}
        for idx in segments.keys():
            result_stenosis[idx] = {}
            result_stenosis[idx]['分段编号'] = [segments[idx][i].get('分段编号', np.nan) for i in range(len(segments[idx]))]
            result_stenosis[idx]['狭窄程度'] = [segments[idx][i].get('狭窄程度', np.nan) for i in range(len(segments[idx]))]  
            # 拼接为dict
            result_stenosis[idx] = dict(zip(result_stenosis[idx]['分段编号'], result_stenosis[idx]['狭窄程度']))

        # 提取key中的数字，并转换为字符串
        result_stenosis_new = {}
        for idx in result_stenosis.keys():
            result_stenosis_new[idx] = {}
            for key in result_stenosis[idx].keys():
                key_ = re.findall(value_pattern, key)
                if len(key_) > 0:
                    key_ = key_[0]
                    result_stenosis_new[idx].update({key_: result_stenosis[idx][key]})
                else:
                    pass
        result_stenosis = result_stenosis_new

        # 把所有value转换为数字
        for idx in result_stenosis.keys():
            for key in result_stenosis[idx].keys():
                v = re.findall(value_pattern, str(result_stenosis[idx][key]))
                if len(v) > 0:
                    result_stenosis[idx][key] = float(v[0])
                else:
                    result_stenosis[idx][key] = result_stenosis[idx][key]
        

        for idx in result_stenosis.keys():
            for key in result_stenosis[idx].keys():
                # 如果包含中文则打印
                if re.search(r'[\u4e00-\u9fa5]', str(key)):
                    print(idx, key)
                # 如果不是字符串而是数字，则打印
                # if not isinstance(key, str):
                #     print(idx, key)
                # 如果包含非数字字符串，则打印
                if not str(key).isdigit():
                    print(idx, key)

        stage_cadras = {}
        sis = {}  # 有多少条血管的"斑块类型"不为"无"或者""
        for idx in result_stenosis.keys():
            if 2622255 == idx:
                print(result_stenosis[idx])
            # 获取所有的斑块类型
            type_of_plaque = [segments[idx][i].get('斑块类型', '') for i in range(len(segments[idx]))]
            # 用re.findall找到type_of_plaque中有多少个无或者''
            # 匹配 "无" 或空字符串
            pattern = r'^"无"$|^$'
            # 计算匹配的数量
            count = sum(1 for item in type_of_plaque if re.match(pattern, item))
            sis[idx] = len(type_of_plaque) - count

            # 一个患者只要有任何一条血管的狭窄程度是1，就是CADRADS 5
            if 1 in result_stenosis[idx].values():
                stage_cadras[idx] = '5'
                print(f'{idx} is CADRADS 5')
            # 一个患者只要有编号为5的血管的狭窄程度是大于等于0.5，或者((1 or 2or 3) and (6 or 7 or 8) and (11 or 13))都是大于等于0.7，就是CADRADS 4b
            elif result_stenosis[idx].get('5', 0) >= 0.5 or (
                (result_stenosis[idx].get('1', 0) >= 0.7 or result_stenosis[idx].get('2', 0) >= 0.7 or result_stenosis[idx].get('3', 0) >= 0.7) and
                (result_stenosis[idx].get('6', 0) >= 0.7 or result_stenosis[idx].get('7', 0) >= 0.7 or result_stenosis[idx].get('8', 0) >= 0.7) and
                (result_stenosis[idx].get('11', 0) >= 0.7 or result_stenosis[idx].get('13', 0) >= 0.7)
            ):
                stage_cadras[idx] = '4b'
                print(f'{idx} is CADRADS 4b')
            # 一个患者只要有任何一条或两条血管的狭窄程度在0.7-0.99之间，就是CADRADS 4a
            elif any([0.7 <= value < 1 for value in result_stenosis[idx].values()]):
                stage_cadras[idx] = '4a'
                print(f'{idx} is CADRADS 4a')
            # 一个患者只要有任何一条血管的狭窄程度在0.5-0.69之间，就是CADRADS 3
            elif any([0.5 <= value < 0.7 for value in result_stenosis[idx].values()]):
                stage_cadras[idx] = '3'
                print(f'{idx} is CADRADS 3')
            # 一个患者只要有任何一条血管的狭窄程度在0.25-0.49之间，就是CADRADS 2
            elif any([0.25 <= value < 0.5 for value in result_stenosis[idx].values()]):
                stage_cadras[idx] = '2'
                print(f'{idx} is CADRADS 2')
            # 一个患者只要有任何一条血管的狭窄程度在0.1-0.24之间，就是CADRADS 1
            elif any([0.1 <= value < 0.25 for value in result_stenosis[idx].values()]):
                stage_cadras[idx] = '1'
                print(f'{idx} is CADRADS 1')
            # 如果有任何一条血管有斑块，但是管腔无明显狭窄，也记做CAD RADS-1
            elif sis[idx] > 0:
                stage_cadras[idx] = '1'
                print(f'{idx} is CADRADS 1')
            # 一个患者所有血管的狭窄程度都是0，就是CADRADS 0
            elif all([value == 0 for value in result_stenosis[idx].values()]):
                stage_cadras[idx] = '0'
                print(f'{idx} is CADRADS 0')
            # 如果以上都不满足，则是CADRADS N
            else:
                stage_cadras[idx] = 'N'
                print(f'{idx} is CADRADS N')

        # 打印多少个CADRADS分级
        print(pd.Series(stage_cadras).value_counts())

        # 根据sis计算p分级：sis=0,为p0， 小于等于2为p1，sis3-4为p2，sis5-7为p3，sis大于等于8为p4
        stage_p = {}
        for idx in sis.keys():
            if sis[idx] == 0:
                stage_p[idx] = 'P0'
                print(f'{idx} is P0')
            elif 1 <= sis[idx] <= 2:
                stage_p[idx] = 'P1'
                print(f'{idx} is P1')
            elif 3 <= sis[idx] <= 4:
                stage_p[idx] = 'P2'
                print(f'{idx} is P2')
            elif 5 <= sis[idx] <= 7:
                stage_p[idx] = 'P3'
                print(f'{idx} is P3')
            elif sis[idx] >= 8:
                stage_p[idx] = 'P4'
                print(f'{idx} is P4')

        print(pd.Series(stage_p).value_counts())

        # 提取心肌桥信息
        myocardial_bridge = self.extract_myocardial_bridge(segments)

        # 提取非钙化斑块信息
        noncalcified_plaque = self.extract_noncalcified_plaque(segments)

        return stage_cadras, sis, stage_p, myocardial_bridge, noncalcified_plaque

    # 写一个函数提取心肌桥信息
    def extract_myocardial_bridge(self, segments):
        """ 提取心肌桥：只要任何一条血管的key"是存在心肌桥"有"有"或"是"，就是心肌桥阳性
        """
        myocardial_bridge = {}
        for idx in segments.keys():
            # 提取所有的"是存在心肌桥"的值
            if 2544937 == idx:
                print(segments[idx])
            # 提取所有的"是存在心肌桥"的key和value
            pattern = re.compile(r'^"有"$|^"是"$')
            bridge = [
                segment.get('是存在心肌桥', segment.get('是否存在心肌桥', ''))
                for segment in segments[idx]
            ]
            # 检查是否存在心肌桥
            has_myocardial_bridge = any(pattern.match(item) for item in bridge)

            myocardial_bridge[idx] = has_myocardial_bridge
        
        # print多少心肌桥阳性
        print(pd.Series(myocardial_bridge).value_counts())

        return myocardial_bridge
    
    # 写一个函数提取是否有“非钙化斑块”
    def extract_noncalcified_plaque(self, segments):
        """ 提取非钙化斑块：只要任何一条血管的key"斑块类型"有"非钙化"，就是非钙化斑块阳性
        """
        noncalcified_plaque = {}
        for idx in segments.keys():
            # 提取所有的"斑块类型"的值
            if 2544937 == idx:
                print(segments[idx])
            # 提取所有的"斑块类型"的key和value
            pattern = re.compile(r'非钙化')
            plaque = [
                segment.get('斑块类型', '')
                for segment in segments[idx]
            ]
            # 检查是否存在非钙化斑块
            is_positive = any(pattern.search(item) for item in plaque)

            noncalcified_plaque[idx] = is_positive
        
        # print多少非钙化斑块阳性
        print(pd.Series(noncalcified_plaque).value_counts())

        return noncalcified_plaque

    def main(self, file):
        """
        主函数
        :param file: 数据文件
        :return: 提取出的信息
        """
        data = self.read_data(file)
        segments, coronary_artery_dominance, coronary_artery_abnormal = self.extract(data)
        stage_cadras, sis, stage_p, myocardial_bridge, noncalcified_plaque = self.get_stage(segments)
        result = {
            'coronary_artery_dominance': coronary_artery_dominance,
            'coronary_artery_abnormal': coronary_artery_abnormal,
            'stage_cadras': stage_cadras,
            'sis': sis,
            'stage_p': stage_p,
            'myocardial_bridge': myocardial_bridge,
            'noncalcified_plaque': noncalcified_plaque
        }
        # to pd.DataFrame
        result = pd.DataFrame(result)
        return result



if __name__ == '__main__':
    file = '../data/GPT4O输出结果汇总1012.xlsx'
    process = Process()
    results = process.main(file)
    # save to excel,给index列命名为检查号
    results.to_excel('../data/results.xlsx', index_label='检查号', engine='openpyxl')
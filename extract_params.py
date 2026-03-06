import pandas as pd
import json
import os
import logging

# 設置 logger - 按需初始化
from analysis.logging_config import get_logger
logger = get_logger("SSS.System")

def extract_params_from_csv(csv_file, trial_numbers):
    """從CSV文件中提取指定trial_number的參數"""
    df = pd.read_csv(csv_file)
    results = {}
    
    for trial_num in trial_numbers:
        row = df[df['trial_number'] == trial_num]
        if not row.empty:
            params_str = row.iloc[0]['parameters']
            strategy = row.iloc[0]['strategy']
            data_source = row.iloc[0]['data_source']
            
            # 解析JSON字符串
            params = json.loads(params_str)
            
            # 添加策略類型和數據源
            params['strategy_type'] = strategy
            params['smaa_source'] = data_source
            
            results[trial_num] = params
    
    return results

def generate_param_presets():
    """生成param_presets字典條目"""
    
    # 定義需要提取的trial_number和對應的名稱
    trial_mappings = {
        # RMA策略
        'RMA_2414_69': {'file': 'RMA_Factor_TWII__2414.TW_fine_grained_processed.csv', 'trial': 69},
        'RMA_2414_72': {'file': 'RMA_Factor_TWII__2414.TW_fine_grained_processed.csv', 'trial': 72},
        'RMA_Self_669': {'file': 'RMA_Self_fine_grained_processed.csv', 'trial': 669},
        'RMA_Self_914': {'file': 'RMA_Self_fine_grained_processed.csv', 'trial': 914},
        
        # Single策略
        'single_Self_1587': {'file': 'single_Self_fine_grained_processed.csv', 'trial': 1587},
        'single_Self_1887': {'file': 'single_Self_fine_grained_processed.csv', 'trial': 1887},
        'single_Self_906': {'file': 'single_Self_fine_grained_processed.csv', 'trial': 906},
        
        # SSMA Turn策略
        'ssma_turn_2414_273': {'file': 'ssma_turn_Factor_TWII__2414.TW_fine_grained_processed.csv', 'trial': 273},
        'ssma_turn_2414_550': {'file': 'ssma_turn_Factor_TWII__2414.TW_fine_grained_processed.csv', 'trial': 550},
        'ssma_turn_2414_811': {'file': 'ssma_turn_Factor_TWII__2414.TW_fine_grained_processed.csv', 'trial': 811},
    }
    
    csv_dir = "results/fine_grained_processed"
    all_params = {}
    
    for name, info in trial_mappings.items():
        csv_file = os.path.join(csv_dir, info['file'])
        trial_num = info['trial']
        
        if os.path.exists(csv_file):
            try:
                params = extract_params_from_csv(csv_file, [trial_num])
                if trial_num in params:
                    all_params[name] = params[trial_num]
                    logger.info(f"成功提取 {name}: trial_number {trial_num}")
                else:
                    logger.warning(f"在 {csv_file} 中未找到 trial_number {trial_num}")
            except Exception as e:
                logger.error(f"處理 {csv_file} 時出錯: {e}")
        else:
            logger.error(f"文件不存在 {csv_file}")
    
    return all_params

def format_param_presets(params):
    """格式化參數為Python字典格式"""
    formatted = {}
    
    for name, param_dict in params.items():
        # 構建參數字符串
        param_str = "{"
        for key, value in param_dict.items():
            if isinstance(value, str):
                param_str += f'"{key}": "{value}", '
            else:
                param_str += f'"{key}": {value}, '
        param_str = param_str.rstrip(', ') + "}"
        
        formatted[name] = param_str
    
    return formatted

if __name__ == "__main__":
    logger.info("開始提取參數...")
    params = generate_param_presets()
    
    if params:
        logger.info("提取的參數:")
        formatted = format_param_presets(params)
        
        logger.info("# 生成的param_presets條目:")
        for name, param_str in formatted.items():
            logger.info(f'"{name}": {param_str},')
        
        logger.info("# 完整的param_presets字典:")
        logger.info("param_presets = {")
        for name, param_str in formatted.items():
            logger.info(f'    "{name}": {param_str},')
        logger.info("}")
    else:
        logger.warning("未找到任何參數") 
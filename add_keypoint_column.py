import pandas as pd

def add_keypoint_column(input_file, output_file):
    """
    为CSV文件添加关键点列
    
    规则：
    - 按井号（转换后JH）分组
    - 当status_predict从0变为1时，关键点=1
    - 当status_predict从1变为2时，关键点=2
    - 当status_predict从2变为3时，关键点=3
    - 其他情况关键点=0
    """
    
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 添加关键点列，初始值为0
    df['关键点'] = 0
    
    # 按井号分组处理
    for well_name, group in df.groupby('转换后JH'):
        print(f"处理井号: {well_name}")
        
        # 获取该井号的数据索引
        well_indices = group.index.tolist()
        
        # 获取status_predict序列
        status_predict = group['status_predict'].values
        
        # 找到状态变化的关键点
        for i in range(1, len(status_predict)):
            prev_status = status_predict[i-1]
            curr_status = status_predict[i]
            
            # 检查状态变化
            if prev_status == 0 and curr_status == 1:
                # 从直井段变为造斜段
                df.loc[well_indices[i], '关键点'] = 1
            elif prev_status == 1 and curr_status == 2:
                # 从造斜段变为稳斜段
                df.loc[well_indices[i], '关键点'] = 2
            elif prev_status == 2 and curr_status == 3:
                # 从稳斜段变为降斜段
                df.loc[well_indices[i], '关键点'] = 3
    
    # 保存结果（带BOM的UTF-8格式）
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"处理完成，结果保存到: {output_file}")
    
    # 显示统计信息
    print("\n关键点列统计:")
    keypoint_counts = df['关键点'].value_counts().sort_index()
    print(keypoint_counts)
    
    # 显示各井号的关键点统计
    print("\n各井号关键点统计:")
    for well_name, group in df.groupby('转换后JH'):
        keypoint_stats = group['关键点'].value_counts().sort_index()
        if keypoint_stats.sum() > 0:  # 只显示有关键点的井
            print(f"  {well_name}: {dict(keypoint_stats)}")
    
    return df

if __name__ == "__main__":
    # 处理文件
    input_file = "validation_without_label_predictions.csv"
    output_file = "validation_with_keypoints.csv"
    
    try:
        result_df = add_keypoint_column(input_file, output_file)
        print(f"\n成功处理 {len(result_df)} 行数据")
        
        # 显示前几行结果
        print("\n前20行结果:")
        print(result_df[['序号', '转换后JH', 'JX', 'dy➗dx', 'status_predict', '关键点']].head(20))
        
        # 显示有关键点的行
        keypoint_rows = result_df[result_df['关键点'] > 0]
        if len(keypoint_rows) > 0:
            print(f"\n找到 {len(keypoint_rows)} 个关键点:")
            print(keypoint_rows[['序号', '转换后JH', 'JX', 'dy➗dx', 'status_predict', '关键点']])
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

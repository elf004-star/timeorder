import pandas as pd

def add_status_column(input_file, output_file):
    """
    为CSV文件添加status列
    
    规则：
    - 按井号（转换后JH）分组
    - 在关键点为1之前，status = 0
    - 在关键点为1之后，遇到2之前，status = 1
    - 在关键点为2之后，遇到3之前，status = 2
    - 在关键点为3之后，status = 3
    """
    
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 添加status列，初始值为0
    df['status'] = 0
    
    # 按井号分组处理
    for well_name, group in df.groupby('转换后JH'):
        print(f"处理井号: {well_name}")
        
        # 获取该井号的数据索引
        well_indices = group.index.tolist()
        
        # 找到关键点的位置
        key_points = group['关键点'].values
        
        # 初始化状态
        current_status = 0
        
        for i, key_point in enumerate(key_points):
            if key_point == 1:
                current_status = 1
            elif key_point == 2:
                current_status = 2
            elif key_point == 3:
                current_status = 3
            
            # 设置当前行的status
            df.loc[well_indices[i], 'status'] = current_status
    
    # 保存结果（带BOM的UTF-8格式）
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"处理完成，结果保存到: {output_file}")
    
    # 显示统计信息
    print("\nStatus列统计:")
    print(df['status'].value_counts().sort_index())
    
    return df

if __name__ == "__main__":
    # 处理文件
    input_file = "data.csv"
    output_file = "data_with_status.csv"
    
    try:
        result_df = add_status_column(input_file, output_file)
        print(f"\n成功处理 {len(result_df)} 行数据")
        
        # 显示前几行结果
        print("\n前10行结果:")
        print(result_df[['序号', '转换后JH', '关键点', 'status']].head(10))
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

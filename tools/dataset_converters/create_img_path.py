import json
import os
import argparse

def generate_image_paths(output_file: str = 'image_paths.json', 
                         start_id: str = '2022101693',
                         start_idx: int = 1,
                         count: int = 6000,
                         base_dir: str = '/data/upload/AI22/2022101693/') -> None:
    """生成标准格式的图像路径JSON文件"""
    items = []
    
    # 生成指定数量的图像信息
    for i in range(count):
        # 将起始ID作为字符串处理，直接追加序号
        item_id = f"{start_id}{i + 1}"
        
        item = {
            "id": int(item_id),  # 转换回整数
            "data": {
                "image": f"{base_dir}{start_idx + i:04d}.jpg"
            },
            "annotations": [],
            "predictions": []
        }
        items.append(item)
    
    # 写入标准JSON文件（带逗号分隔）
    with open(output_file, 'w') as f:
        json.dump(items, f, indent=2)
    
    print(f"已生成包含{len(items)}个图像信息的JSON文件: {output_file}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成图像路径JSON文件')
    parser.add_argument('--output', default='image_paths.json', help='输出JSON文件路径')
    parser.add_argument('--start_id', type=str, default='2022101693', help='起始ID（字符串形式）')
    parser.add_argument('--start_idx', type=int, default=1, help='起始图像编号')
    parser.add_argument('--count', type=int, default=6000, help='生成图像数量')
    parser.add_argument('--base_dir', default='/data/upload/AI22/2022101693/', help='图像基础路径')
    
    args = parser.parse_args()
    
    # 生成JSON文件
    generate_image_paths(
        output_file=args.output,
        start_id=args.start_id,
        start_idx=args.start_idx,
        count=args.count,
        base_dir=args.base_dir
    )    
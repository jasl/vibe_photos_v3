#!/usr/bin/env python3
"""
数据集采样脚本
用于从30,000+张照片中智能采样，生成PoC1测试集
"""

import os
import random
import shutil
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple

class DatasetSampler:
    """数据集采样器"""
    
    def __init__(self, source_dir: str, target_dir: str):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.metadata = {
            'source_path': str(source_dir),
            'target_path': str(target_dir),
            'created_at': datetime.now().isoformat(),
            'samples': []
        }
    
    def analyze_dataset(self) -> Dict:
        """分析数据集结构"""
        print("📊 分析数据集...")
        
        stats = {
            'total_dirs': 0,
            'total_files': 0,
            'by_year': defaultdict(int),
            'by_location': defaultdict(int),
            'file_types': defaultdict(int),
            'size_distribution': []
        }
        
        for dir_path in self.source_dir.iterdir():
            if not dir_path.is_dir():
                continue
            
            stats['total_dirs'] += 1
            
            # 解析目录名（如：Beijing, October 29, 2025）
            dir_name = dir_path.name
            year = self._extract_year(dir_name)
            location = self._extract_location(dir_name)
            
            if year:
                stats['by_year'][year] += 1
            if location:
                stats['by_location'][location] += 1
            
            # 统计文件
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    stats['total_files'] += 1
                    ext = file_path.suffix.lower()
                    stats['file_types'][ext] += 1
                    
                    # 记录文件大小
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    stats['size_distribution'].append(size_mb)
        
        return stats
    
    def _extract_year(self, dir_name: str) -> str:
        """从目录名提取年份"""
        import re
        match = re.search(r'20\d{2}', dir_name)
        return match.group() if match else None
    
    def _extract_location(self, dir_name: str) -> str:
        """从目录名提取地点"""
        # 格式：地点, 日期 或 只有日期
        parts = dir_name.split(',')
        if len(parts) > 1:
            return parts[0].strip()
        return None
    
    def stratified_sample(self, 
                          total_samples: int = 1000,
                          strategy: str = 'balanced') -> List[Path]:
        """
        分层采样
        
        策略：
        - balanced: 按年份均匀采样
        - recent: 偏向最近的照片
        - random: 完全随机
        """
        print(f"🎲 执行{strategy}采样，目标：{total_samples}张")
        
        all_images = []
        year_groups = defaultdict(list)
        
        # 收集所有图片并按年份分组
        for dir_path in self.source_dir.iterdir():
            if not dir_path.is_dir():
                continue
            
            year = self._extract_year(dir_path.name) or "unknown"
            
            for file_path in dir_path.iterdir():
                if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.heic']:
                    all_images.append(file_path)
                    year_groups[year].append(file_path)
        
        # 根据策略采样
        if strategy == 'balanced':
            # 每年均匀采样
            samples = []
            years = sorted(year_groups.keys())
            samples_per_year = total_samples // len(years)
            
            for year in years:
                year_images = year_groups[year]
                n = min(samples_per_year, len(year_images))
                samples.extend(random.sample(year_images, n))
            
            # 补充不足的样本
            if len(samples) < total_samples:
                remaining = total_samples - len(samples)
                pool = [img for img in all_images if img not in samples]
                samples.extend(random.sample(pool, min(remaining, len(pool))))
        
        elif strategy == 'recent':
            # 偏向最近的照片（70%最近3年，30%其他）
            recent_years = ['2023', '2024', '2025']
            recent_images = []
            older_images = []
            
            for year, images in year_groups.items():
                if year in recent_years:
                    recent_images.extend(images)
                else:
                    older_images.extend(images)
            
            recent_count = int(total_samples * 0.7)
            older_count = total_samples - recent_count
            
            samples = []
            if recent_images:
                samples.extend(random.sample(recent_images, 
                              min(recent_count, len(recent_images))))
            if older_images:
                samples.extend(random.sample(older_images, 
                              min(older_count, len(older_images))))
        
        else:  # random
            # 完全随机
            samples = random.sample(all_images, min(total_samples, len(all_images)))
        
        return samples[:total_samples]
    
    def create_sample_dataset(self, 
                             samples: List[Path],
                             preserve_structure: bool = True):
        """创建采样数据集"""
        print(f"📁 创建采样数据集：{len(samples)}个文件")
        
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        for i, source_file in enumerate(samples, 1):
            if preserve_structure:
                # 保持原目录结构
                rel_dir = source_file.parent.name
                target_subdir = self.target_dir / rel_dir
                target_subdir.mkdir(exist_ok=True)
                target_file = target_subdir / source_file.name
            else:
                # 扁平化结构
                target_file = self.target_dir / f"{i:04d}_{source_file.name}"
            
            # 复制文件（或创建符号链接节省空间）
            if not target_file.exists():
                # 使用符号链接节省空间
                target_file.symlink_to(source_file)
                # 或者复制：shutil.copy2(source_file, target_file)
            
            # 记录元数据
            self.metadata['samples'].append({
                'index': i,
                'source': str(source_file),
                'target': str(target_file),
                'size_mb': source_file.stat().st_size / (1024 * 1024)
            })
            
            if i % 100 == 0:
                print(f"  已处理 {i}/{len(samples)} 文件...")
        
        # 保存元数据
        metadata_file = self.target_dir / 'dataset_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 采样完成！元数据保存至：{metadata_file}")
    
    def create_test_sets(self):
        """创建多个测试集"""
        print("🔬 创建测试集...")
        
        # Phase 1: 小规模验证集（1000张）
        print("\n--- Phase 1: 验证集 ---")
        validation_samples = self.stratified_sample(1000, 'balanced')
        validation_dir = self.target_dir / 'phase1_validation'
        sampler = DatasetSampler(self.source_dir, validation_dir)
        sampler.create_sample_dataset(validation_samples)
        
        # Phase 2: 性能测试集（5000张）
        print("\n--- Phase 2: 性能测试集 ---")
        performance_samples = self.stratified_sample(5000, 'recent')
        performance_dir = self.target_dir / 'phase2_performance'
        sampler = DatasetSampler(self.source_dir, performance_dir)
        sampler.create_sample_dataset(performance_samples)
        
        # Phase 3: 准备全量处理
        print("\n--- Phase 3: 全量数据集 ---")
        print(f"全量数据集路径：{self.source_dir}")
        print(f"包含 30,000+ 张照片，400GB 数据")
        print("建议使用增量处理和断点续传功能")
    
    def generate_report(self):
        """生成数据集分析报告"""
        stats = self.analyze_dataset()
        
        report = f"""
# 数据集分析报告

## 概览
- 总目录数：{stats['total_dirs']:,}
- 总文件数：{stats['total_files']:,}

## 年份分布
"""
        for year in sorted(stats['by_year'].keys()):
            count = stats['by_year'][year]
            report += f"- {year}: {count} 个目录\n"
        
        report += "\n## 地点分布（Top 10）\n"
        locations = sorted(stats['by_location'].items(), key=lambda x: x[1], reverse=True)[:10]
        for location, count in locations:
            report += f"- {location}: {count} 个目录\n"
        
        report += "\n## 文件类型\n"
        for ext, count in stats['file_types'].items():
            report += f"- {ext}: {count:,} 个文件\n"
        
        if stats['size_distribution']:
            avg_size = sum(stats['size_distribution']) / len(stats['size_distribution'])
            max_size = max(stats['size_distribution'])
            min_size = min(stats['size_distribution'])
            
            report += f"""
## 文件大小
- 平均: {avg_size:.1f} MB
- 最大: {max_size:.1f} MB
- 最小: {min_size:.1f} MB
"""
        
        return report

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据集采样工具')
    parser.add_argument('--source', default='/Users/jasl/Workspaces/exported_photos',
                       help='源数据目录')
    parser.add_argument('--target', default='./test_datasets',
                       help='目标目录')
    parser.add_argument('--samples', type=int, default=1000,
                       help='采样数量')
    parser.add_argument('--strategy', choices=['balanced', 'recent', 'random'],
                       default='balanced', help='采样策略')
    parser.add_argument('--analyze-only', action='store_true',
                       help='仅分析数据集')
    
    args = parser.parse_args()
    
    sampler = DatasetSampler(args.source, args.target)
    
    if args.analyze_only:
        # 仅分析
        report = sampler.generate_report()
        print(report)
        
        # 保存报告
        report_file = Path(args.target) / 'dataset_analysis.md'
        report_file.parent.mkdir(exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n报告已保存至：{report_file}")
    else:
        # 执行采样
        samples = sampler.stratified_sample(args.samples, args.strategy)
        sampler.create_sample_dataset(samples)
        
        # 生成报告
        report = sampler.generate_report()
        report_file = Path(args.target) / 'sampling_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

if __name__ == "__main__":
    main()

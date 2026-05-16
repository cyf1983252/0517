"""
labeling — 增强型用户属性打标模块

基于多字段语义分析、逐级匹配、改进决策树的打标器。
标签生成独立于模型训练，每个用户的标签只依赖其自身资料。
"""
from .labeler import label_user, label_dataframe

# 算法示例

## 使用指南
1. 按 'CTRL + P'打开命令行面板，输入 "terminal: Create New Terminal" 打开一个命令行终端 或者按
   ' CTRL + ` '直接打开命令行终端。
2. 在命令行里输入 `pip install -r requirements.txt` 按 `ENTER` 安装示例程序所需依赖库。
   如果安装报错 可以尝试以下命令升级pip
   ```
   pip install --upgrade pip
   pip install --upgrade setuptools
   pip install ez_setup
   ```
3. 在命令行里输入 'cd 1_算法示例' 并按 `ENTER` 进入"算法示例"目录。
4. 算法示例所用的数据是WN18数据集的一个子集,包含实体id集，关系id集和训练集。保存在txt文档中，元素格式
   分别为'实体\t对应id'，'关系\t对应id'，'头实体\t尾实体\t关系'。
5. 在命令行里输入 `python solution.py` 按 `ENTER` 运行示例程序。

## 运行结果
   正确运行代码后，会在example_result目录下生成文件entity_50dim_batch40和文件relation_50dim_batch40，里面存放训练好的实体向量和关系向量。

## 备注
   如果是在命令行上运行代码，需要按照使用指南的顺序进入**正确的目录**才可运行成功。








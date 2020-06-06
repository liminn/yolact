# 说明
该仓库代码，针对用于牛听听的卡片字符拼接

# 文件结构
joint_cards_20200606：
 - test_images: 包含测试图片、测试txt以及测试结果图
 - joint_cards.py: JointCards类
 - demo.py: 示例文件
 - 运行：`python3 demo.py --txt xx/test_images/test_1.txt --image xx/test_images/test_1.jpg`

# test case
对于"test_1.txt":
- 输出：[[0, 1], [2, 3]]
对于"test_2.txt":
- 输出：[[0, 1, 2, 3, 4]]
对于"test_3.txt":
- 输出：[[2], [0, 1], [4, 5], [3]]
对于"test_4.txt":
- 输出：[[2, 3], [1, 0]]
对于"test_5.txt":
- 输出：[[0], [1, 2]]
对于"test_6.txt":
- 输出：[[0, 1], [2]]
对于"test_7.txt":
- 输出：[[0], [2, 1]]
对于"test_8.txt":
- 输出：[[0, 1]]
对于"test_9.txt":
- 输出：[[0], [1]]
对于"test_10.txt":
- 输出：[[0], [2, 1], [3]]
对于"test_11.txt":
- 输出：[[4, 3, 2, 1, 0]]


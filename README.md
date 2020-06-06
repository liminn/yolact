# Joint Card
This repo is for the task of joint cards

# Prerequisites
- numpy
- opencv

# File structure
joint_cards_20200606：
 - test_images: contains test images, test txts and test results
 - joint_cards.py: the script that defines the JointCards class
 - demo.py: demo script

# Usage of Demo
run:
`python3 demo.py --txt xx/test_images/test_1.txt --image xx/test_images/test_1.jpg`

# Test cases
for `python3 demo.py --txt xx/test_images/test_1.txt --image xx/test_images/test_1.jpg`
- output: [[0, 1], [2, 3]]

for `python3 demo.py --txt xx/test_images/test_2.txt --image xx/test_images/test_2.jpg`
- output: [[0, 1, 2, 3, 4]]

for `python3 demo.py --txt xx/test_images/test_3.txt --image xx/test_images/test_3.jpg`
- output: [[2], [0, 1], [4, 5], [3]]

for `python3 demo.py --txt xx/test_images/test_4.txt --image xx/test_images/test_4.jpg`
- output: [[2, 3], [1, 0]]

for `python3 demo.py --txt xx/test_images/test_5.txt --image xx/test_images/test_5.jpg`
- output: [[0], [1, 2]]

for `python3 demo.py --txt xx/test_images/test_6.txt --image xx/test_images/test_6.jpg`
- output: [[0, 1], [2]]

for `python3 demo.py --txt xx/test_images/test_7.txt --image xx/test_images/test_7.jpg`
- output: [[0], [2, 1]]

for `python3 demo.py --txt xx/test_images/test_8.txt --image xx/test_images/test_8.jpg`
- output: [[0, 1]]

for `python3 demo.py --txt xx/test_images/test_9.txt --image xx/test_images/test_9.jpg`
- output: [[0], [1]]

for `python3 demo.py --txt xx/test_images/test_10.txt --image xx/test_images/test_10.jpg`
- output: [[0], [2, 1], [3]]

for `python3 demo.py --txt xx/test_images/test_11.txt --image xx/test_images/test_11.jpg`
- output: [[4, 3, 2, 1, 0]]


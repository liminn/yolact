# Joint Cards
This repo is for the task of joint cards

# Prerequisites
- numpy
- opencv

# File structure
joint_cards：
 - test_resources: contains test images, test txts and test results
 - joint_cards.py: the script that defines the algorithm of joint-cards
 - demo.py: demo script

# Usage of Demo
run: `python3 demo.py --path xx/test_resources`

# Test cases
run : `python3 demo.py --path xx/test_resources`

correct output:
```Python
input: test_1.txt
output: {'index': [[0, 1], [2, 3]], 'rectangle': [[281, 121, 612, 121, 612, 303, 281, 303], [475, 270, 735, 270, 735, 429, 475, 429]], 'text': ['RM', 'Yp']}

input: test_10.txt
output: {'index': [[0], [2, 1], [3]], 'rectangle': [[651, 95, 851, 95, 851, 276, 651, 276], [584, 244, 872, 244, 872, 432, 584, 432], [554, 415, 701, 415, 701, 513, 554, 513]], 'text': ['i', 'NN', 'm']}

input: test_11.txt
output: {'index': [[4, 3, 2, 1, 0]], 'rectangle': [[471, 311, 1035, 311, 1035, 492, 471, 492]], 'text': ['siqID']}

input: test_2.txt
output: {'index': [[0, 1, 2, 3, 4]], 'rectangle': [[309, 142, 985, 142, 985, 309, 309, 309]], 'text': ['HrdNg']}

input: test_3.txt
output: {'index': [[2], [0, 1], [4, 5], [3]], 'rectangle': [[704, 41, 857, 41, 857, 176, 704, 176], [618, 167, 871, 167, 871, 404, 618, 404], [19, 453, 117, 453, 117, 525, 19, 525], [64, 508, 147, 508, 147, 552, 64, 552]], 'text': ['V', 'fa', 'oy', 'B']}

input: test_4.txt
output: {'index': [[2, 3], [1, 0]], 'rectangle': [[1041, 326, 1229, 326, 1229, 476, 1041, 476], [243, 348, 410, 348, 410, 479, 243, 479]], 'text': ['Zz', 'UX']}

input: test_5.txt
output: {'index': [[0], [1, 2]], 'rectangle': [[583, 233, 737, 233, 737, 354, 583, 354], [546, 389, 811, 389, 811, 468, 546, 468]], 'text': ['h', 'Zb']}

input: test_6.txt
output: {'index': [[0, 1], [2]], 'rectangle': [[204, 580, 352, 580, 352, 639, 204, 639], [391, 638, 477, 638, 477, 672, 391, 672]], 'text': ['ip', 'p']}

input: test_7.txt
output: {'index': [[0], [2, 1]], 'rectangle': [[136, 223, 260, 223, 260, 325, 136, 325], [105, 327, 243, 327, 243, 453, 105, 453]], 'text': ['I', 'uY']}

input: test_8.txt
output: {'index': [[0, 1]], 'rectangle': [[619, 337, 863, 337, 863, 429, 619, 429]], 'text': ['lN']}

input: test_9.txt
output: {'index': [[0], [1]], 'rectangle': [[970, 266, 1116, 266, 1116, 383, 970, 383], [172, 582, 247, 582, 247, 620, 172, 620]], 'text': ['l', 'r']}
```

- PS: you can check `test_resources/text_*_result.jpg` to confirm the results.

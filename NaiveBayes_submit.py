from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import label
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
import json
import os
import random

import warnings

#numpy开源库的一个bug，需要ignore这个warning
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

#通过编号找题目，这是函数是提取字符串中的数字
def safe_int(num):
    try:
        return int(num)
    except ValueError:
        result = []
        for c in num:
            if not ('0' <= c <= '9'):
                break
            result.append(c)
        if len(result) == 0:
            return 0
        return int(''.join(result))

# 和上面相反，这个函数通过提取字符串中的题目（不要数字）
def safe_name(num):
    result = []
    for c in num:
        if ('0' <= c <= '9'):
            continue
        result.append(c)
    if len(result) == 0:
        return 0
    return (''.join(result))


#每一类题目用一个list存储
algorithm = []
array = []
basic = []
binarytree = []
bitoperation = []
classtemplate = []
constructor = []
deconstructor = []
dynamicprogramming = []
exception = []
functionoverload = []
functiontemplate = []
inheritance = []
linkedlist = []
macro = []
memory = []
operatoroverload = []
pointer = []
polymorphism = []
queue = []
recursion = []
sort = []
stack = []
stl = []
struct = []

rootpath = "D:/软件工程学习/专业课/大三/实训——题目分类与推荐/题库"
tagpath = "D:/软件工程学习/专业课/大三/实训——题目分类与推荐/data/tags/"

# 打开已经有标签的题目，加入题号和标签，以准备训练用
f = open(tagpath+"/algorithm.json", 'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    algorithm.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath+"/array.json",'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    array.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath+"/basic.json",'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    basic.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath+"/binary tree.json",'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    binarytree.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath+"/bit operation.json",'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    bitoperation.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath+"/class template.json",'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    classtemplate.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath+"/constructor.json",'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    constructor.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath+"/deconstructor.json",'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    deconstructor.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath+"/dynamic programming.json",'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    dynamicprogramming.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath+"/exception.json", 'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    exception.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath+"/function overload.json",'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    functionoverload.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath+"/function template.json",'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    functiontemplate.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath+"/inheritance.json",'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    inheritance.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath+"/linked list.json",'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    linkedlist.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath+"/macro.json",'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    macro.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath+"/memory.json",'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    memory.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath+"/operator overload.json",'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    operatoroverload.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath+"/pointer.json",'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    pointer.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath + "/polymorphism.json",'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    polymorphism.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath+"/queue.json",'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    queue.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath+"/recursion.json",'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    recursion.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath+"/sort.json",'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    sort.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath+"/stack.json",'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    stack.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath+"/STL.json",'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    stl.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

f = open(tagpath+"/struct.json",'r', encoding='utf-8')
text = json.loads(f.read())
for i in range(len(text['problems'])):
    struct.append((text['problems'][i]['id'],text['problems'][i]['tags']))
f.close()

# 获取数据集，读取题号，标签，题面，题目标题
def get_dataset():
    data = []
    # 遍历题目所在的文件夹
    for root, dirs, files in os.walk(rootpath):
        for name in dirs:
            for id,tags in algorithm:
                if (safe_int(name) == int(id)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), tags, id,safe_name(name)))
            for id,tags in array:
                if (safe_int(name) == int(id)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), tags, id,safe_name(name)))

            for bid,btags in basic:
                if (safe_int(name) == int(bid)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), btags, bid, safe_name(name)))

            for btid,btags in binarytree:
                if (safe_int(name) == int(btid)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), btags, btid, safe_name(name)))

            for boid,botags in bitoperation:
                if (safe_int(name) == int(boid)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), botags, boid, safe_name(name)))

            for ctid,ctags in classtemplate:
                if (safe_int(name) == int(ctid)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), ctags, ctid, safe_name(name)))

            for conid,cotags in constructor:
                if (safe_int(name) == int(conid)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), cotags, conid, safe_name(name)))

            for deconid,detags in deconstructor:
                if (safe_int(name) == int(deconid)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), detags, deconid, safe_name(name)))

            for dpid,dptags in dynamicprogramming:
                if (safe_int(name) == int(dpid)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), dptags, dpid, safe_name(name)))

            for excid,exctags in exception:
                if (safe_int(name) == int(excid)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), exctags, excid, safe_name(name)))

            for foid,fotags in functionoverload:
                if (safe_int(name) == int(foid)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), fotags, foid, safe_name(name)))

            for ftid,ftags in functiontemplate:
                if (safe_int(name) == int(ftid)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), ftags, ftid, safe_name(name)))

            for inhid,intags in inheritance:
                if (safe_int(name) == int(inhid)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), intags, inhid, safe_name(name)))

            for llid,lltags in linkedlist:
                if (safe_int(name) == int(llid)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), lltags, llid, safe_name(name)))

            for mid,mtags in macro:
                if (safe_int(name) == int(mid)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), mtags, mid, safe_name(name)))

            for mmid,mmtags in memory:
                if (safe_int(name) == int(mmid)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), mmtags, mmid, safe_name(name)))

            for ooid,ootags in operatoroverload:
                if (safe_int(name) == int(ooid)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), ootags, ooid, safe_name(name)))

            for pid,ptags in pointer:
                if (safe_int(name) == int(pid)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), ptags, pid, safe_name(name)))

            for pid,ptags in polymorphism:
                if (safe_int(name) == int(pid)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), ptags, pid, safe_name(name)))

            for rid,rtags in recursion:
                if (safe_int(name) == int(rid)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), rtags, rid, safe_name(name)))

            for qid,qtags in queue:
                if (safe_int(name) == int(qid)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), qtags, qid, safe_name(name)))

            for sid,stags in sort:
                if (safe_int(name) == int(sid)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), stags, sid, safe_name(name)))

            for stid,stags in stack:
                if (safe_int(name) == int(stid)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), stags, stid, safe_name(name)))

            for sid,stags in stl:
                if (safe_int(name) == int(sid)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), stags, sid, safe_name(name)))

            for strid,strtags in struct:
                if (safe_int(name) == int(strid)):
                    with open(os.path.join(root, name, 'Description.md'), 'r', encoding='utf-8') as f:
                        data.append((f.read(), strtags, strid, safe_name(name)))

    random.shuffle(data)
    # for i in data:
    #   print(i)
    return data

# 数据集的获取
data = get_dataset()

# 训练和测试，把一半的题目拿来做训练集，一半的题目进行预测
def train_and_test_data(data_):
    filesize = int(0.5 * len(data_))
    # 训练集和测试集的比例为5:5
    train_data_ = [each[0] for each in data_[:filesize]]
    train_target_ = [each[1] for each in data_[:filesize]]

    train_id_ = [each[2] for each in data_[:filesize]]
    train_name_ = [each[3] for each in data_[:filesize]]

    test_data_ = [each[0] for each in data_[filesize:]]
    test_target_ = [each[1] for each in data_[filesize:]]
    test_id_ = [each[2] for each in data_[filesize:]]
    test_name_ = [each[3] for each in data_[filesize:]]

    return train_data_, train_target_, train_id_, train_name_, test_data_, test_target_, test_id_, test_name_


train_data, train_target, train_id, train_name, test_data, test_target, test_id, test_name = train_and_test_data(data)

# nb分词器，使用朴素贝叶斯算法
nbc = Pipeline([
    ('vect', TfidfVectorizer(
        analyzer='word', token_pattern=r'\w{1,}',ngram_range = (0,5), max_features=5000
    )),
    ('clf', MultinomialNB(alpha=1.0)),
])

#多标签分类时，需要先转化成int型数组，这里的encoder.fit其实是把文本标准化为数组，得到词向量
encoder = preprocessing.LabelEncoder()
train_target_y = encoder.fit_transform(train_target)
test_target_y = encoder.fit_transform(test_target)

# 训练多项式模型NB分类器
nbc.fit(train_data, train_target_y)

predict = nbc.predict(test_data)  # 在测试集上预测结果

count = 0
for left, right in zip(predict, test_target_y):
    if left == right:
        #统计正确个数
        count = count + 1
#统计正确率
#print(count/len(test_target))

# 序列化输出，需要把题目封装成dict的形式，才能序列化地导入json文件

printdata = []
# 输出训练集
for id, taglist, name in zip(train_id, train_target, train_name):
    #规定字典的格式
    dict = {'id': 'Zara', 'name': 'sgenlien', 'tags': 7};
    dict['id'] = id
    dict['name'] = name
    dict['tags'] = taglist
    #将字典加入printdata，之后序列化输出json
    printdata.append(dict)

# 使用inverse_transform函数，可以把数组转回原来的文本形式
printtaglist = []
printtaglist = encoder.inverse_transform(predict).all()

#输出预测集
for id, taglist, name in zip(test_id, printtaglist, test_name):
    dict = {'id': 'Zara', 'name': 'sgenlien', 'tags': 7};
    dict['id'] = id
    dict['name'] = name
    dict['tags'] = taglist
    printdata.append(dict)

# 按格式输出到json文件
with open("problem.json", "w", encoding="UTF-8") as f_dump:
    s_dump = json.dump(printdata, f_dump, ensure_ascii=False,sort_keys=True, separators=(',', ':'))
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time

from tensorflow.contrib import rnn

class get_stock_data():
    """
    获得股票分类数据，用以深度学习。格式为按照时间顺序，由小到大排序的：
    [开盘价, 最高价, 最低价, 收盘价，换手率，成交量, 流通市值, 真实收益, 日期]
    真实收益定义：下一个交易日开盘买入，再下一个交易日开盘卖出所得到的收益，算法为 100 * (open_next_next / open_next - 1)。
    > 如果真实收益的回归预测效果一般，可以再试试分类预测：真实收益>= n 和 <n 等
    """
    def __init__(self,):
        self.file_paths = {}

    def get_train_test_data_new(self, batch_size, file_path, time_step = 60):
        """获取训练集和测试集数据"""
        self_variable_name = "data_cur_{0}".format(file_path)
        if self_variable_name in self.file_paths:
            cursor = self.file_paths[self_variable_name]
        else:
            self.file_paths[self_variable_name] = cursor = pd.read_csv(file_path, iterator = True)

        data_temp = cursor.get_chunk(batch_size)
        #input_x
        data_temp['0'] = data_temp['0'].apply(eval)
        #input_y
        data_temp['1'] = data_temp['1'].apply(eval)
        #input_y列表有3个元素：收益率、日期、股票代码, 只获取收益率
        data_temp["1"] = data_temp["1"].apply(lambda x: x[0])

        #整成矩阵形式返回
        return np.array(data_temp["0"].tolist()).reshape(batch_size , time_step * 7), np.array(data_temp["1"].tolist()).reshape(batch_size , 1)
    
    def get_train_test_data_softmax(self, batch_size, file_path, time_step = 60):
        """
        将收益变成分类问题。分类定义在classify函数内，分别为：
            0: <= 0%
            1: (0%, 5%)
            2: [5%, 以上)
        """

        self_variable_name = "data_cur_{0}".format(file_path)
        if self_variable_name in self.file_paths:
            cursor = self.file_paths[self_variable_name]
        else:
            self.file_paths[self_variable_name] = cursor = pd.read_csv(file_path, iterator = True)

        data_temp = cursor.get_chunk(batch_size)
        data_temp['0'] = data_temp['0'].apply(eval)
        data_temp['1'] = data_temp['1'].apply(eval)
        data_temp["1"] = data_temp["1"].apply(lambda x: x[0])

        def classify(y):
            if y <= 0:
                return [1,0,0]
            elif y >= 5:
                return [0,0,1]
            else:
                return [0,1,0]

        data_temp["1"] = data_temp["1"].apply(lambda x: classify(x))
        return np.array(data_temp["0"].tolist()).reshape(batch_size , time_step * 7), np.array(data_temp["1"].tolist()).reshape(batch_size , 3)

class Deeplearing():
    """
    """
    def stock_lstm(self,):
        """
        使用LSTM处理股票数据
        直接预测收益
        """
        #获取当前python文件运行在哪个目录下并去掉最后的文件名，如：F:/deeplearning/main.py --> F:/deeplearning
        #在linux下同样起作用
        basic_path = os.path.dirname(os.path.abspath(__file__))
        #定义存储模型的文件路径
        #os.path.join(basic_path, "stock_rnn_save.ckpt")表示在运行的python文件路径下保存，文件名为stock_rnn.ckpt，在我环境下运行总是提示“另一个程序正在使用此文件，进程无法访问”，换到其他路径就OK
        model_save_path = r"F:\\123\\save\\stock_rnn_save.ckpt"#os.path.join(basic_path, "stock_rnn_save.ckpt")
        #定义训练集的文件路径，当前为运行的python文件路径下，文件名为train_data.csv
        train_csv_path = os.path.join(basic_path, "train_data.csv")
        #定义测试集的文件路径，当前为运行的python文件路径下，文件名为test_data.csv
        test_csv_path = os.path.join(basic_path, "test_data.csv")
        #学习率
        learning_rate = 0.001
        #喂数据给LSTM的原始数据有几行，即：一次希望LSTM能“看到”多少个交易日的数据
        origin_data_row = 60
        #喂给LSTM的原始数据有几列，即：日线数据有几个元素
        origin_data_col = 7
        #LSTM网络有几层
        layer_num = 2
        #LSTM网络，每层有几个神经元
        cell_num = 256
        #最后输出的数据维度，即：要预测几个数据，该处只预测收益率，只有一个数据
        output_num = 1
        #每次给LSTM网络喂多少行经过处理的股票数据。该参数依据自己显卡和网络大小动态调整，越大 一次处理的就越多，越能占用更多的计算资源
        batch_size = tf.placeholder(tf.int32, [])
        #输入层、输出层权重、偏置。
        #通过这两对参数，LSTM层能够匹配输入和输出的数据
        W = {
         'in':tf.Variable(tf.truncated_normal([origin_data_col, cell_num], stddev = 1), dtype = tf.float32),
         'out':tf.Variable(tf.truncated_normal([cell_num, output_num], stddev = 1), dtype = tf.float32)
         }
        bias = {
        'in':tf.Variable(tf.constant(0.1, shape=[cell_num,]), dtype = tf.float32),
        'out':tf.Variable(tf.constant(0.1, shape=[output_num,]), dtype = tf.float32)
        }
        #告诉LSTM网络，即将要喂的数据是几行几列
        #None的意思就是喂数据时，行数不确定交给tf自动匹配
        #我们喂得数据行数其实就是batch_size，但是因为None这个位置tf只接受数字变量，而batch_size是placeholder定义的Tensor变量，表示我们在喂数据的时候才会告诉tf具体的值是多少
        input_x = tf.placeholder(tf.float32, [None, origin_data_col * origin_data_row])
        input_y = tf.placeholder(tf.float32, [None, output_num])
        #处理过拟合问题。该值在其起作用的层上，给该层每一个神经元添加一个“开关”，“开关”打开的概率是keep_prob定义的值，一旦开关被关了，这个神经元的输出将被“阻断”。这样做可以平衡各个神经元起作用的重要性，杜绝某一个神经元“一家独大”，各种大佬都证明这种方法可以有效减弱过拟合的风险。
        keep_prob = tf.placeholder(tf.float32, [])

        #通过reshape将输入的input_x转化成2维，-1表示函数自己判断该是多少行，列必须是origin_data_col
        #转化成2维 是因为即将要做矩阵乘法，矩阵一般都是2维的（反正我没见过3维的）
        input_x_after_reshape_2 = tf.reshape(input_x, [-1, origin_data_col])

        #当前计算的这一行，就是输入层。输入层的激活函数是relu,并且施加一个“开关”，其打开的概率为keep_prob
        #input_rnn即是输入层的输出，也是下一层--LSTM层的输入
        input_rnn = tf.nn.dropout(tf.nn.relu_layer(input_x_after_reshape_2, W['in'], bias['in']), keep_prob)
        
        #通过reshape将输入的input_rnn转化成3维
        #转化成3维，是因为即将要进入LSTM层，接收3个维度的数据。粗糙点说，即LSTM接受：batch_size个，origin_data_row行cell_num列的矩阵，这里写-1的原因与input_x写None一致
        input_rnn = tf.reshape(input_rnn, [-1, origin_data_row, cell_num])

        #定义一个带着“开关”的LSTM单层，一般管它叫细胞
        def lstm_cell():
            cell = rnn.LSTMCell(cell_num, reuse = tf.get_variable_scope().reuse)
            return rnn.DropoutWrapper(cell, output_keep_prob = keep_prob)
        #这一行就是tensorflow定义多层LSTM网络的代码
        lstm_layers = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple = True)
        #初始化LSTM网络
        init_state = lstm_layers.zero_state(batch_size, dtype = tf.float32)

        #使用dynamic_rnn函数，告知tf构建多层LSTM网络，并定义该层的输出
        outputs, state = tf.nn.dynamic_rnn(lstm_layers, inputs = input_rnn, initial_state = init_state, time_major = False)
        h_state = state[-1][1]

        #该行代码表示了输出层
        #将LSTM层的输出，输入到输出层，输出层最终得出的值即为预测的收益
        y_pre = tf.matmul(h_state, W['out']) + bias['out']

        #损失函数，用作指导tf
        #loss的定义为：(使用预测的收益 - 喂给LSTM的这60个交易日对应的真实收益)的平方，如果有多个（即：batch_size个且batch_size大于1），那就求一下平均值（先挨个求平方，再整个求平均值）
        loss = tf.reduce_mean(tf.square(tf.subtract(y_pre, input_y)))
        #告诉tf，它需要做的事情就是就是尽可能将loss减小
        #learning_rate是减小的这个过程中的参数。如果将我们的目标比喻为“从北向南走路走到菜市场”，我理解的是
        #learning_rate越大，我们走的每一步就迈的越大。初看似乎步子越大越好，但是其实并不能保证每一步都是向南走
        #的，有可能因为训练数据的原因，导致我们朝西走了一大步。或者我们马上就要到菜市场了，但是一大步走过去，给
        #走过了。。。综上，这个learning_rate（学习率参数）的取值，无法给出一个比较普适的，还是需要根据实际情况去
        #尝试和调整。0.001的取值是tf给的默认值
        #上述例子是个人理解用尽可能通俗易懂地语言表达。如有错误，欢迎指正
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        #这块定义了一个新的值，用作展示训练的效果
        #它的定义为：选择预测值和实际值差别最大的情况并将差值返回
        accuracy = tf.reduce_max(tf.abs(tf.subtract(y_pre, input_y)))
        
        #tf要求必须如此定义一个init变量，用以在初始化运行（也就是没有保存模型）时加载各个变量
        init = tf.global_variables_initializer()
        
        #用以保存参数的函数（跑完下次再跑，就可以直接读取上次跑的结果而不必重头开始）
        saver = tf.train.Saver()
        #获取数据（这是我们自己定义的类）
        data_get = get_stock_data()
        #设置GPU按需增长，在多个GPU时，能更多地占用资源
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #使用with，保证执行完后正常关闭tf
        with tf.Session(config = config) as sess:
            try:
                #定义了存储模型的文件路径，即：当前运行的python文件路径下，文件名为stock_rnn.ckpt
                saver.restore(sess, model_save_path)
                print ("成功加载模型参数")
            except:
                #如果是第一次运行，通过init告知tf加载并初始化变量
                print ("未加载模型参数，文件被删除或者第一次运行")
                sess.run(init)
            
            #给batch_size赋值
            _batch_size = 200
            for i in range(20000):
                try:
                    #读取训练集数据
                    train_x, train_y = data_get.get_train_test_data_new(batch_size = _batch_size, file_path = train_csv_path)
                except StopIteration:
                    print ("训练集均已训练完毕")
                    train_accuracy = sess.run(accuracy, feed_dict={
                        input_x:train_x, input_y: train_y, keep_prob: 1.0, batch_size: _batch_size})
                    print ("step: {0}, training_accuracy: {1}".format(i + 1, train_accuracy))
                    saver.save(sess, model_save_path)
                    print("保存模型\n")
                    break

                if (i + 1) % 20 == 0:
                    train_accuracy = sess.run(accuracy, feed_dict={
                        input_x:train_x, input_y: train_y, keep_prob: 1.0, batch_size: _batch_size})
                    #输出
                    print ("step: {0}, training_accuracy: {1}".format(i + 1, train_accuracy))
                    saver.save(sess, model_save_path)
                    print("保存模型\n")
                    ############################################
                    #这部分代码作用为：每次保存模型，顺便将预测收益和真实收益输出保存至show_y_pre.txt文件下。熟悉tf可视化，完全可以采用可视化替代
                    _y_pre_train = sess.run(y_pre, feed_dict={
                        input_x: train_x, input_y: train_y, keep_prob: 1.0, batch_size:  _batch_size})
                    _loss = sess.run(loss, feed_dict={
                        input_x:train_x, input_y: train_y, keep_prob: 1.0, batch_size: _batch_size})
                    a1 = np.array(train_y).reshape(1, _batch_size)
                    b1 = np.array(_y_pre_train).reshape(1, _batch_size)
                    with open (os.path.join(basic_path, "show_y_pre.txt"), "w") as f:
                        f.write(str(a1.tolist()))
                        f.write("\n")
                        f.write(str(b1.tolist()))
                        f.write("\n")
                        f.write(str(_loss))
                    ############################################
                #按照给定的参数训练一次LSTM神经网络
                sess.run(train_op, feed_dict={input_x: train_x, input_y: train_y, keep_prob: 0.6, batch_size: _batch_size})

            #计算测试数据的准确率
            #读取测试集数据
            test_size = 100
            test_x, test_y = data_get.get_train_test_data_new(batch_size = test_size, file_path = test_csv_path)
            print ("test accuracy {0}".format(sess.run(accuracy, feed_dict={
                input_x: test_x, input_y: test_y, keep_prob: 1.0, batch_size:test_size})))

    def stock_lstm_softmax(self,):
        """
        使用LSTM处理股票数据
        分类预测
        """
        #获取当前python文件运行在哪个目录下并去掉最后的文件名，如：F:/deeplearning/main.py --> F:/deeplearning
        #在linux下同样起作用
        basic_path = os.path.dirname(os.path.abspath(__file__))
        #定义存储模型的文件路径，当前为运行的python文件路径下，文件名为stock_rnn.ckpt
        model_save_path = r"F:\\123\\save\\stock_rnn_save.ckpt"#os.path.join(basic_path, "stock_rnn.ckpt")
        #定义训练集的文件路径，当前为运行的python文件路径下，文件名为train_data.csv
        train_csv_path = os.path.join(basic_path, "train_data.csv")
        #定义测试集的文件路径，当前为运行的python文件路径下，文件名为test_data.csv
        test_csv_path = os.path.join(basic_path, "test_data.csv")
        #学习率
        learning_rate = 0.001
        #喂数据给LSTM的原始数据有几行，即：一次希望LSTM能“看到”多少个交易日的数据
        origin_data_row = 60
        #喂给LSTM的原始数据有几列，即：日线数据有几个元素
        origin_data_col = 7
        #LSTM网络有几层
        layer_num = 2
        #LSTM网络，每层有几个神经元
        cell_num = 200
        #最后输出的数据维度，即：要预测几个数据，该处需要处理分类问题，按照自己设定的类型数量设定
        output_num = 3
        #每次给LSTM网络喂多少行经过处理的股票数据。该参数依据自己显卡和网络大小动态调整，越大 一次处理的就越多，越能占用更多的计算资源
        batch_size = tf.placeholder(tf.int32, [])
        #输入层、输出层权重、偏置。
        #通过这两对参数，LSTM层能够匹配输入和输出的数据
        W = {
         'in':tf.Variable(tf.truncated_normal([origin_data_col, cell_num], stddev = 1), dtype = tf.float32),
         'out':tf.Variable(tf.truncated_normal([cell_num, output_num], stddev = 1), dtype = tf.float32)
         }
        bias = {
        'in':tf.Variable(tf.constant(0.1, shape=[cell_num,]), dtype = tf.float32),
        'out':tf.Variable(tf.constant(0.1, shape=[output_num,]), dtype = tf.float32)
        }
        #告诉LSTM网络，即将要喂的数据是几行几列
        #None的意思就是喂数据时，行数不确定交给tf自动匹配
        #我们喂得数据行数其实就是batch_size，但是因为None这个位置tf只接受数字变量，而batch_size是placeholder定义的Tensor变量，表示我们在喂数据的时候才会告诉tf具体的值是多少
        input_x = tf.placeholder(tf.float32, [None, origin_data_col * origin_data_row])
        input_y = tf.placeholder(tf.float32, [None, output_num])
        #处理过拟合问题。该值在其起作用的层上，给该层每一个神经元添加一个“开关”，“开关”打开的概率是keep_prob定义的值，一旦开关被关了，这个神经元的输出将被“阻断”。这样做可以平衡各个神经元起作用的重要性，杜绝某一个神经元“一家独大”，各种大佬都证明这种方法可以有效减弱过拟合的风险。
        keep_prob = tf.placeholder(tf.float32, [])

        #通过reshape将输入的input_x转化成2维，-1表示函数自己判断该是多少行，列必须是origin_data_col
        #转化成2维 是因为即将要做矩阵乘法，矩阵一般都是2维的（反正我没见过3维的）
        input_x_after_reshape_2 = tf.reshape(input_x, [-1, origin_data_col])

        #当前计算的这一行，就是输入层。输入层的激活函数是relu,并且施加一个“开关”，其打开的概率为keep_prob
        #input_rnn即是输入层的输出，也是下一层--LSTM层的输入
        input_rnn = tf.nn.dropout(tf.nn.relu_layer(input_x_after_reshape_2, W['in'], bias['in']), keep_prob)
        
        #通过reshape将输入的input_rnn转化成3维
        #转化成3维，是因为即将要进入LSTM层，接收3个维度的数据。粗糙点说，即LSTM接受：batch_size个，origin_data_row行cell_num列的矩阵，这里写-1的原因与input_x写None一致
        input_rnn = tf.reshape(input_rnn, [-1, origin_data_row, cell_num])

        #定义一个带着“开关”的LSTM单层，一般管它叫细胞
        def lstm_cell():
            cell = rnn.LSTMCell(cell_num, reuse=tf.get_variable_scope().reuse)
            return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        #这一行就是tensorflow定义多层LSTM网络的代码
        lstm_layers = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple = True)
        #初始化LSTM网络
        init_state = lstm_layers.zero_state(batch_size, dtype = tf.float32)

        #使用dynamic_rnn函数，告知tf构建多层LSTM网络，并定义该层的输出
        outputs, state = tf.nn.dynamic_rnn(lstm_layers, inputs = input_rnn, initial_state = init_state, time_major = False)
        h_state = state[-1][1]

        #该行代码表示了输出层
        #将LSTM层的输出，输入到输出层（输出层带softmax激活函数），输出为各个分类的概率
        #假设有3个分类，那么输出举例为：[0.001, 0.992, 0.007]，表示第1种分类概率千分之1，第二种99.2%, 第三种千分之7
        y_pre = tf.nn.softmax(tf.matmul(h_state, W['out']) + bias['out'])

        #损失函数，用作指导tf
        #loss定义为交叉熵损失函数，softmax输出层大多都使用的这个损失函数。关于该损失函数详情可以百度下
        loss = -tf.reduce_mean(input_y * tf.log(y_pre))
        #告诉tf，它需要做的事情就是就是尽可能将loss减小
        #learning_rate是减小的这个过程中的参数。如果将我们的目标比喻为“从北向南走路走到菜市场”，我理解的是
        #learning_rate越大，我们走的每一步就迈的越大。初看似乎步子越大越好，但是其实并不能保证每一步都是向南走
        #的，有可能因为训练数据的原因，导致我们朝西走了一大步。或者我们马上就要到菜市场了，但是一大步走过去，给
        #走过了。。。综上，这个learning_rate（学习率参数）的取值，无法给出一个比较普适的，还是需要根据实际情况去
        #尝试和调整。0.001的取值是tf给的默认值
        #上述例子是个人理解用尽可能通俗易懂地语言表达。如有错误，欢迎指正
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        #这块定义了一个新的值，用作展示训练的效果
        #它的定义为：预测对的 / 总预测数，例如：0.55表示预测正确了55%
        correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        #用以保存参数的函数（跑完下次再跑，就可以直接读取上次跑的结果而不必重头开始）
        saver = tf.train.Saver(tf.global_variables())
        
        #tf要求必须如此定义一个init变量，用以在初始化运行（也就是没有保存模型）时加载各个变量
        init = tf.global_variables_initializer()
        #获取数据（这是我们自己定义的类）
        data_get = get_stock_data()
        #设置GPU按需增长，在多个GPU时，能更多地占用资源
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        #使用with，保证执行完后正常关闭tf
        with tf.Session(config = config) as sess:
            try:
                #定义了存储模型的文件路径，即：当前运行的python文件路径下，文件名为stock_rnn.ckpt
                saver.restore(sess, model_save_path)
                print ("成功加载模型参数")
            except:
                #如果是第一次运行，通过init告知tf加载并初始化变量
                print ("未加载模型参数，文件被删除或者第一次运行")
                sess.run(init)
            
            #给batch_size赋值
            _batch_size = 200
            for i in range(20000):
                try:
                    #读取训练集数据
                    train_x, train_y = data_get.get_train_test_data_softmax(batch_size = _batch_size, file_path = train_csv_path)
                except StopIteration:
                    print ("训练集均已训练完毕")
                    train_accuracy = sess.run(accuracy, feed_dict={
                        input_x: train_x, input_y: train_y, keep_prob: 1.0, batch_size: _batch_size})
                    print ("step: {0}, training_accuracy: {1}".format(i + 1, train_accuracy))
                    saver.save(sess, model_save_path)
                    print("保存模型\n")
                    break

                if (i + 1) % 20 == 0:
                    train_accuracy = sess.run(accuracy, feed_dict={
                        input_x:train_x, input_y: train_y, keep_prob: 1.0, batch_size: _batch_size})
                    print ("step: {0}, training_accuracy: {1}".format(i + 1, train_accuracy))
                    saver.save(sess, model_save_path)
                    print("保存模型\n")
                #按照给定的参数训练一次LSTM神经网络
                sess.run(train_op, feed_dict={input_x: train_x, input_y: train_y, keep_prob: 0.6, batch_size: _batch_size})

            #计算测试数据的准确率
            #读取测试集数据
            test_size = 100
            test_x, test_y = data_get.get_train_test_data_softmax(batch_size = _batch_size, file_path = test_csv_path)
            print ("test accuracy {0}".format(sess.run(accuracy, feed_dict={
                input_x: test_x, input_y: test_y, keep_prob: 1.0, batch_size:_batch_size})))

if __name__ == "__main__":
    pass
    t = Deeplearing()
    #t.stock_lstm()
    t.stock_lstm_softmax()

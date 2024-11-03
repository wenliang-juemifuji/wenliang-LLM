from transformer import Transformer
import tensorflow_datasets as tfds

# 定义优化器
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# 数据编码
def encode(lang1, lang2):
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(lang1.numpy()) +
             [tokenizer_pt.vocab_size+1]
    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
             lang2.numpy()) + [tokenizer_en.vocab_size+1]
    return lang1, lang2

def filter_long_sent(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)

# 将Python运算转换为TensorFlow运算
def tf_encode(pt, en):
    return tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
 

def loss_fun(y_ture, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_ture, 0))  # 为0掩码标1
        loss_ = loss_object(y_ture, y_pred)
    
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

# 构建掩码
def create_mask(inputs,targets):
    encode_padding_mask = create_padding_mark(inputs)
    # 这个掩码用于遮挡输入解码层第二层的编码层输出
    decode_padding_mask = create_padding_mark(inputs)

    # look_ahead掩码，遮挡未预测的词
    look_ahead_mask = create_look_ahead_mark(tf.shape(targets)[1])
    # 解码层的第一层得到padding掩码
    decode_targets_padding_mask = create_padding_mark(targets)

    # 合并解码层的第一层掩码
    combine_mask = tf.maximum(decode_targets_padding_mask, look_ahead_mask)

    return encode_padding_mask, combine_mask, decode_padding_mask

@tf.function
def train_step(inputs, targets):
    target_inp = targets[:,:-1]
    target_real = targets[:,1:]
    # 构造掩码
    encode_padding_mask, combined_mask, decode_padding_mask = create_mask(inputs, target_inp)
    # 训练过程
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inputs, tar_inp, True, encode_padding_mask,
                                     combined_mask, decode_padding_mask)
        loss = loss_fun(tar_real, predictions)
    # 求梯度
    gradients = tape.gradient(loss, transformer.trainable_variables)
    # 反向传播
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    # 记录损失值和准确率
    train_loss(loss)
    train_accuracy(tar_real, predictions)

def predict_func(inp_sentence):
    start_token = [tokenizer_pt.vocab_size]
    end_token = [tokenizer_pt.vocab_size + 1]  
    # 输入语句是葡萄牙语，增加开始和结束标记
    inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)
    
    # 因为目标是英文，输入Transformer的第一个词应该是英文的开始标记。
    decoder_input = [tokenizer_en.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):       
        enc_padding_mask, combined_mask, dec_padding_mask = create_mask(
            encoder_input, output)
        predictions, attention_weights = transformer(encoder_input, output, False,
            enc_padding_mask, combined_mask, dec_padding_mask)

        # 从seq_len维度选择最后一个词
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # 如果predicted_id等于结束标记，就返回结果
        if predicted_id == tokenizer_en.vocab_size + 1:
            return tf.squeeze(output, axis=0), attention_weights

        # 连接predicted_id与输出，作为解码器的输入传递到解码器
    output = tf.concat([output, predicted_id], axis=-1)
    
    return tf.squeeze(output, axis=0), attention_weights 

# 翻译输出
def translate(sentence, plot=""):
    result, attention_weights = predict_func(sentence)
    predicted_sentence = tokenizer_en.decode([i for i in result 
                                              if i < tokenizer_en.vocab_size])  

    print(‘输入: {}’.format(sentence))
    print(‘预测输出: {}’.format(predicted_sentence))


def main():

    examples, metadata = tfds.load("ted_hrlr_translate/pt_to_en", with_info=True,
                                    as_supervised=True)
    for pt, en in train_examples:
        print(pt.numpy().decode(‘utf-8’))
        print(en.numpy().decode(‘utf-8’))
        break
    
    # 将数据转换为subwords格式
    train_examples, val_examples = examples["train"], examples["validation"]
    tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)
    tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)

    # 过滤长度超过MAX_LENGTH的数据
    MAX_LENGTH=40
       
    # 构建数据集
    BUFFER_SIZE = 20000
    BATCH_SIZE = 64

    
    # 使用.map()执行相关图操作
    train_dataset = train_examples.map(tf_encode)
    # 过滤过长的数据
    train_dataset = train_dataset.filter(filter_long_sent)
    # 使用缓存数据加速读入
    train_dataset = train_dataset.cache()
    # 打乱并获取批数据
    train_dataset = train_dataset.padded_batch(
    BATCH_SIZE, padded_shapes=([40], [40]))  
    # 设置训练数据
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    # 构建验证集数据
    val_dataset = val_examples.map(tf_encode)
    val_dataset = val_dataset.filter(filter_long_sent).padded_batch(
    BATCH_SIZE, padded_shapes=([40], [40]))
    de_batch, en_batch = next(iter(train_dataset))

    # 设置模型超参数
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    input_vocab_size = tokenizer_pt.vocab_size + 2
    target_vocab_size = tokenizer_en.vocab_size + 2
    max_seq_len = 40
    dropout_rate = 0.1

    learing_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learing_rate, beta_1=0.9, 
                                         beta_2=0.98, epsilon=1e-9)
    # 定义目标函数
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

        # 创建Transformer模型
    transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size,
                              target_vocab_size, max_seq_len, dropout_rate)

    # 创建检查点管理器
    checkpoint_path = "./checkpoint/train"
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    # 检查点管理器
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)
    
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print(‘last checkpoit restore’)

    EPOCHS = 1
    step_list = []
    loss_list = []
    step = 0
    for epoch in range(EPOCHS):
        start = time.time()
        # 重置记录项
        train_loss.reset_states()
        train_accuracy.reset_states()
        # inputs为葡萄牙语，targets为英文
        for batch, (inputs, targets) in enumerate(train_dataset):
            # 训练
            train_step(inputs, targets)
            if batch % 500 == 0:
                loss = train_loss.result()
                step_list.append(step)
                loss_list.append(loss)
            step += 1
        if (epoch + 1) % 2 == 0:
            ckpt_save_path = ckpt_manager.save()
    
    #打印损失值
    plt.plot(step_list, loss_list)
    plt.xlabel("train step")
    plt.ylabel("loss")

    translate("este é um problema que temos que resolver.")
    print ("真实输出: this is a problem we have to solve .\n")

if __name__ == "__main__":
    main()

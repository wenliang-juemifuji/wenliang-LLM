from transformer import Transformer
import tensorflow_datasets as tfds



def main():
  examples, metadata = tfds.load(‘ted_hrlr_translate/pt_to_en’, with_info=True,
	                                as_supervised=True)
	for pt, en in train_examples:
	    print(pt.numpy().decode(‘utf-8’))
	    print(en.numpy().decode(‘utf-8’))
	    break
    
  # 将数据转换为subwords格式
	train_examples, val_examples = examples[‘train’], examples[‘validation’]
	tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
	    (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)
	tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
	    (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)

if __name__ == "__main__":
    main()

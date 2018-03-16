import tensorflow as tf
import sys
sys.path.extend(['/home/ec2-user/environment/tensorflow'])
f = open("RealMarketPriceDataPT.csv")
data = [item.split(',')[-1].strip() for item in f.readlines()]
print("Data imported successfully")
batch_size = 2
time_steps = 5


price = tf.placeholder(tf.float32, [batch_size])
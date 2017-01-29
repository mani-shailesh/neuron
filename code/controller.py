import models
import util
import data_processor
from tqdm import tqdm

data_store = data_processor.DataStore()
data_dict = data_store.get_data()

batch_size_list = [25, 50, 100, 200, 400]
lr_list = [pow(10, -1 * ii) for ii in range(1, 10)]

num_epochs = 400
lambda_ = 0.001

log_filename = '../results/rmlr_batch_lr.csv'
util.log(log_filename, 'lr,batch_size,epoch#,train_acc,val_acc')

rmlr = models.RMLR(10, '../results/rmlr.csv')

train_X = data_dict['train']['data']
train_Y = data_dict['train']['labels']

val_X = data_dict['val']['data']
val_Y = data_dict['val']['labels']

for lr in tqdm(lr_list):
    for batch_size in tqdm(batch_size_list):
        print("Learning Rate: " + str(lr) + ", Batch Size: " + str(batch_size))
        print("\n---------------------------------------------------------\n")
        train_acc, val_acc, epoch = rmlr.train(train_X, train_Y, lr, batch_size, num_epochs, lambda_,
                                               val_X, val_Y, True)
        util.log(log_filename, str(lr)+","+str(batch_size)+","+str(epoch)+","+str(train_acc)+","+str(val_acc))

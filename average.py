# this program will produce a single matrix which holds
# the average probabilities of the 6 models.

import numpy as np
import pickle
from sklearn.metrics import matthews_corrcoef as mcc
import pandas as pd
import os
import sys
import re


def get_reorder_vec(model_num, original_order_dict, path, q_num, e_num, w_num, d_num, dat_type):
    letters_dict = get_order_of_letters(model_num, path, q_num, e_num, w_num, d_num, dat_type)
    position_dict = {v: k for k, v in letters_dict.items()}
    reorder_dict = {k: original_order_dict[position_dict[k]] for k in range(0, len(letters_dict))}
    return [reorder_dict[i] for i in range(0, len(letters_dict))]


def get_order_of_letters(model_num, path, q_num, e_num, w_num, d_num, dat_type):
    decoder_file = open(
        path + '/model_' + dat_type + '' + q_num + '_e' + e_num + '_m' + str(
            model_num) + '_w' + w_num + '_d' + d_num + '/decoder.pickle', 'rb')
    decoder = pickle.load(decoder_file)
    dict = decoder.word_index
    dict['0'] = 0
    return dict


def sort_matrices_by_letters(amount_of_models, models_matrices, path, q_num, e_num, w_num, d_num, dat_type):
    # sort all letters by the order of model 1.

    num_of_matrices = len(models_matrices[1])
    order_by = get_order_of_letters(1, path, q_num, e_num, w_num, d_num, dat_type)

    for m in range(1, amount_of_models + 1):
        for i in range(num_of_matrices):
            reoredr_vec = get_reorder_vec(m, order_by, path, q_num, e_num, w_num, d_num, dat_type)
            models_matrices[m][i] = models_matrices[m][i][:, reoredr_vec]

    return models_matrices


def sort_pred_matrix(mat, name_vec):
    # creates new matrix out of mat with name vec
    # sorts new matrix
    # return a modified matrix out of the columns of new matrix
    indices = [x for x in range(len(name_vec))]
    name_idx_arr = list(zip(indices, name_vec))

    name_idx_arr.sort(key=lambda x: x[1])
    outputMatrix = []
    for i in range(len(name_vec)):
        outputMatrix.append(mat[name_idx_arr[i][0]])
    outputMatrix = np.array(outputMatrix)
    return outputMatrix


def load_test_pred(idx, path, q_num, e_num, w_num, d_num, dat_type):
    string_to_load = path + '/model_' + dat_type + '' + q_num + '_e' + e_num + '_m' + str(
        idx) + '_w' + w_num + '_d' + d_num + '/out_test_pred.npy'
    return np.load(string_to_load)


def load_name_vec(idx, path, q_num, e_num, w_num, d_num, dat_type):
    alttype = np.dtype([('f0', 'U8'), ('f1', 'U8'), ('f2', 'U8'), ('f3', 'U8')])
    string_to_load = path + '/model_' + dat_type + '' + q_num + '_e' + e_num + '_m' + str(
        idx) + '_w' + w_num + '_d' + d_num + '/out.csv'
    # name_vec = np.genfromtxt(string_to_load, delimiter=',', dtype=alttype, names=True)['pid']
    out_df = pd.read_csv(string_to_load)
    # name_vec = np.genfromtxt(string_to_load, delimiter=',', dtype=alttype, names=True)['pid']
    name_vec = np.array(out_df['pid'])
    return name_vec


def avg_secondary_struct(models_matrices, amount, weights):
    num_of_matrices = len(models_matrices[1])
    num_of_letters = len(models_matrices[1][0])
    num_of_proj = len(models_matrices[1][0][0])

    avg_mat = np.zeros((num_of_matrices, num_of_letters, num_of_proj))

    for i in range(1, amount + 1):
        avg_mat += models_matrices[i] * weights[i]

    return avg_mat / amount


def preform_avg(path, q_num, e_num, w_num, d_num, dat_type, amount):
    # create matrices
    weights = [0, 1, 1, 1, 1, 1, 1]
    matrices = [None]
    name_vecs = [None]
    models_matrices = [None]
    for i in range(1, amount + 1):
        matrices.append(load_test_pred(i, path, q_num, e_num, w_num, d_num, dat_type))
        name_vecs.append(load_name_vec(i, path, q_num, e_num, w_num, d_num, dat_type))
        models_matrices.append(sort_pred_matrix(mat=matrices[i], name_vec=name_vecs[i]))

    # models_matrices = np.array(models_matrices)

    # in this point, all matrices are sorted by the pid from the out file.
    # now we need (?) TODO sort by the letters of the scondary structure
    models_matrices = sort_matrices_by_letters(amount, models_matrices, path, q_num, e_num, w_num, d_num, dat_type)
    avg_mat = avg_secondary_struct(models_matrices, amount, weights)
    # print(np.equal(avg1, avg2))
    output_data(path, q_num, e_num, w_num, d_num, dat_type, avg_mat)


def onehot_to_seq(oh_seq, index, maxlen_seq=700):
    s = ''
    for o in oh_seq:
        i = np.argmax(o)
        if i != 0:
            s += index[i]
        else:
            # s += "_"
            break
    return s


def decode_results(y_, revsere_decoder_index):
    #    print("input     : " + str(x))
    #    print("prediction: " + str(onehot_to_seq(y_, revsere_decoder_index).upper()))
    #    return (str(onehot_to_seq(y_, revsere_decoder_index).upper())[0,len(x)-1])
    return (str(onehot_to_seq(y_, revsere_decoder_index).upper()))


# eager execution accuracy
def ex_accuracy(y_true, y_pred, space_inx):
    y = np.argmax(y_true, axis=- 1)
    y_ = np.argmax(y_pred, axis=- 1)
    mask1 = np.greater(y, 0)
    mask2 = np.not_equal(y, space_inx)
    mask = np.logical_and(mask1, mask2)
    arr = np.equal(y[mask], y_[mask])
    return sum(arr) / float(len(arr))


def ex_accuracy_real(y_true, y_pred, space_inx, by_target=None):
    y = np.argmax(y_true, axis=- 1)
    y_ = np.argmax(y_pred, axis=- 1)
    mask1 = np.greater(y, 0)
    mask2 = np.not_equal(y, space_inx)
    mask = np.logical_and(mask1, mask2)
    arr = np.equal(y[mask], y_[mask])
    arr3 = np.equal(y, y_)
    arr4 = np.logical_and(arr3, mask)
    arr5 = [np.sum(np.max(y_true[i, arr4[i, :]], axis=-1)) for i in range(mask.shape[0])]
    # arr5 = [ sum(np.max(y_pred[i,arr4[i,:]],axis=-1)) for i in range(mask.shape[0])] # sums the probabilities of each target.
    if by_target == None:
        return sum(arr5) / float(len(arr))
    else:
        return arr5 / np.sum(mask, axis=-1)


def confusion_matrix(y_true, y_pred, decoder):
    revsere_decoder_index = {value: key for key, value in decoder.word_index.items()}
    space_inx = decoder.word_index["-"]
    y = np.argmax(y_true, axis=- 1)
    y_ = np.argmax(y_pred, axis=- 1)
    mask1 = np.greater(y, 0)
    mask2 = np.not_equal(y, space_inx)
    mask = np.logical_and(mask1, mask2)

    nclass = len(revsere_decoder_index) + 1
    mat = np.zeros([nclass, nclass], dtype=int)
    ym = y[mask]
    y_m = y_[mask]

    for i in range(len(ym)):
        mat[ym[i], y_m[i]] += 1
    sum_class_true = np.sum(mat, axis=-1)
    sum_class_pred = np.sum(mat, axis=0)
    sum_all = np.sum(sum_class_true)
    acc = np.sum(np.diagonal(mat / sum_all))
    add_epsilon = lambda x: x + 1e-10 if x == 0 else x
    div1 = sum_class_true.reshape(sum_class_true.shape[0], 1)
    div1 = np.apply_along_axis(func1d=add_epsilon, axis=1, arr=div1)
    div2 = sum_class_pred.reshape(1, sum_class_true.shape[0])
    div2 = np.apply_along_axis(func1d=add_epsilon, axis=0, arr=div2)
    recall = np.diagonal(mat / div1)
    precision = np.diagonal(mat / div2)
    add_epsilon = lambda x: np.where(x == 0.0, x + 1e-10, x)
    freq = sum_class_true / sum_all
    div3 = np.apply_along_axis(add_epsilon, axis=0, arr=recall)
    div4 = np.apply_along_axis(add_epsilon, axis=0, arr=precision)

    div5 = (1 / div3 + 1 / div4)
    f_score = 2 / div5
    mat = mat
    m = mcc(ym, y_m)
    return mat, recall, precision, f_score, acc, freq, m


def store_avg_data(out, mat_pred, mat_true, decoder, encoder, test_df):
    revsere_decoder_index = {value: key for key, value in decoder.word_index.items()}
    revsere_encoder_index = {value: key for key, value in encoder.word_index.items()}

    space_inx = decoder.word_index["-"]
    accuracy = ex_accuracy(mat_true, mat_pred, space_inx)
    print(f'average accuracy: {accuracy}')
    decoded_mat_true = []
    decoded_mat_pred = []
    pertarget_performance = []
    ptp = ex_accuracy_real(mat_true, mat_pred, space_inx, by_target=True)
    for i in range(len(mat_pred)):
        decoded_mat_true.append(decode_results(mat_true[i], revsere_decoder_index))
        decoded_mat_pred.append(decode_results(mat_pred[i], revsere_decoder_index))
        pertarget_performance.append(ptp[i])
    out_df = pd.DataFrame()
    out_df["id"] = test_df.id.values
    out_df["old_id"] = test_df.old_id.values
    out_df["pid"] = test_df.pid.values
    out_df["len"] = test_df.len.values
    out_df["input"] = test_df.input.values
    out_df["prediction"] = decoded_mat_pred
    out_df["true"] = decoded_mat_true
    out_df["performance"] = pertarget_performance

    with open(out + "/out.csv", "w") as f:
        out_df.to_csv(f, index=False)

    np.save(out + "/avg_test_pred.npy", mat_pred, allow_pickle=True)
    np.save(out + "/avg_test_true.npy", mat_true, allow_pickle=True)
    mat, recall, precision, f_score, acc, freq, mcc = confusion_matrix(mat_true, mat_pred, decoder)

    q_size = len(revsere_decoder_index) - 2
    with open(out + "/performance" + str(q_size) + ".csv", "w") as f:
        f.write("matthews_corrcoef = " + str(mcc) + "\n")
        f.write(str(accuracy) + "\n")
        f.write(str(mat) + "\n")
        f.write("Class dict: " + str(revsere_decoder_index) + "\n")

        np.set_printoptions(formatter={'float_kind': '{:0.4f}'.format})
        np.set_printoptions(precision=4, suppress=True)

        f.write("*********\n")
        f.write("  | Recall  | Precision   | F_score  | Freq\n")
        for i in range(1, len(revsere_decoder_index) + 1):
            if revsere_decoder_index[i] == '-':
                continue
            f.write("%s | %1.5f |   %1.5f   | %1.5f  | %1.5f \n" % (
                revsere_decoder_index[i], recall[i], precision[i], f_score[i], freq[i]))


def create_out_dir(path):
    dir_path = path + '/averages_model'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def get_true_mat(path, q_num, e_num, w_num, d_num, dat_type):
    true_mat = np.load(
        path + '/model_' + dat_type + '' + q_num + '_e' + e_num + '_m1_w' + w_num + '_d' +
        d_num + '/out_test_true.npy')
    alttype = np.dtype([('f0', 'U8'), ('f1', 'U8'), ('f2', 'U8'), ('f3', 'U8')])
    string_to_load = path + '/model_' + dat_type + '' + q_num + '_e' + e_num + '_m1_w' + w_num + '_d' + d_num + '/out.csv'

    out_df = pd.read_csv(string_to_load)
    # name_vec = np.genfromtxt(string_to_load, delimiter=',', dtype=alttype, names=True)['pid']
    name_vec = np.array(out_df['pid'])
    return sort_pred_matrix(mat=true_mat, name_vec=name_vec)


def get_first_dec_enc(path, q_num, e_num, w_num, d_num, dat_type):
    decoder_file = open(
        path + '/model_' + dat_type + '' + q_num + '_e' + e_num + '_m1_w' + w_num + '_d' + d_num + '/decoder.pickle',
        'rb')

    encoder_file = open(
        path + '/model_' + dat_type + '' + q_num + '_e' + e_num + '_m1_w' + w_num + '_d' + d_num + '/encoder.pickle',
        'rb')
    return pickle.load(decoder_file), pickle.load(encoder_file)


def get_test_df(path, q_num, e_num, w_num, d_num, dat_type):
    df = pd.read_csv(
        path + '/model_' + dat_type + '' + q_num + '_e' + e_num + '_m1_w' + w_num + '_d' + d_num + '/out.csv')
    df = df.sort_values(by=['pid'])
    df = df.rename(index=str, columns={'id': 'old_id', 'pid': 'pid', 'len': 'len'})
    length = df.shape[0]
    new_ids = [i for i in range(1, length + 1)]
    df.insert(loc=0, column='id', value=new_ids)
    #
    # out_df["id"] = test_df.id.values
    # out_df["pid"] = test_df.pid.values
    # out_df["len"] = test_df.len.values
    # out_df["input"] = test_df.input.values

    return df


def output_data(path, q_num, e_num, w_num, d_num, dat_type, avg_mat):
    # given the averaged matrices, now we will produce the files requested.
    # as a way of regularity, all data is set by the order of model 1.

    # create target directory

    out = create_out_dir(path)

    # get test, true data
    true_mat = get_true_mat(path, q_num, e_num, w_num, d_num, dat_type)

    # get decoders and encoders of model 1
    decoder1, encoder1 = get_first_dec_enc(path, q_num, e_num, w_num, d_num, dat_type)

    # get data frame for csv file
    test_df = get_test_df(path, q_num, e_num, w_num, d_num, dat_type)

    # supply outputs to directory
    store_avg_data(out, avg_mat, true_mat, decoder1, encoder1, test_df)


def get_parmas(path):
    list_of_dirs = os.listdir(path)
    for entry in list_of_dirs:
        # Create full path
        fullPath = os.path.join(path, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath) and entry != 'averages_model':
            se = entry.split('_')
            q_num = re.findall(r'\d+', se[1])[0]
            e_num = re.findall(r'\d+', se[2])[0]
            w_num = re.findall(r'\d+', se[4])[0]
            d_num = re.findall(r'\d+', se[5])[0]
            dat_type = re.findall("[a-zA-Z]+", se[1])[0]
            return q_num, e_num, w_num, d_num, dat_type

    else:
        exit(1)


def main():
    n_args = (len(sys.argv))
    if n_args > 1:
        amount_of_models = int(sys.argv[1])
    else:
        amount_of_models = 6
    path = os.path.dirname(os.path.realpath(__file__))
    q_num, e_num, w_num, d_num, dat_type = get_parmas(path)

    preform_avg(path, q_num, e_num, w_num, d_num, dat_type, amount_of_models)
    print("done")
    # tests()


if __name__ == '__main__':
    main()

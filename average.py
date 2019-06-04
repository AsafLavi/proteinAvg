# this program will produce a single matrix which holds
# the average probabilities of the 6 models.

import numpy as np
import pickle
from sklearn.metrics import matthews_corrcoef as mcc
import pandas as pd


def onehot_to_seq(oh_seq, index, maxlen_seq = 700):
    s = ''
    for o in oh_seq:
        i = np.argmax(o)
        if i != 0:
            s += index[i]
        else:
            #s += "_"
            break
    return s


def decode_results(x, y_, revsere_decoder_index):
    #    print("input     : " + str(x))
    #    print("prediction: " + str(onehot_to_seq(y_, revsere_decoder_index).upper()))
    #    return (str(onehot_to_seq(y_, revsere_decoder_index).upper())[0,len(x)-1])
    return (str(onehot_to_seq(y_, revsere_decoder_index).upper()))


def reduceToQ8(mat13, decoder):
    q13 = ['m', 'q', 'p', 'a', 'z', 'l', 'b', 'e', 'g', 'i', 'h', 's', 't', '-']
    q8 = {'0': 0, 'h': 1, 'e': 2, 't': 3, 'c': 4, 's': 5, 'g': 6, 'i': 7, 'b': 8, '-': 9}
    e_group = ['a', 'm', 'q', 'p', 'z', 'e']
    e_inx = [decoder.word_index[w] for w in e_group]
    rdi = {value: key for key, value in decoder.word_index.items()}
    rdi[0] = '0'
    mat8 = np.zeros([10, 10], dtype=int)
    for i in range(mat13.shape[0]):
        for j in range(mat13.shape[1]):
            mi_letter = ''
            if rdi[i] in e_group:
                mi_letter = 'e'  # letter of i
            else:
                mi_letter = rdi[i]
            mj_letter = ''
            if rdi[j] in e_group:
                mj_letter = 'e'  # letter of i
            else:
                mj_letter = rdi[j]
            mat8[q8[mi_letter], q8[mj_letter]] += mat13[i, j]
    mat = mat8
    sum_class_true = np.sum(mat, axis=-1)
    sum_class_pred = np.sum(mat, axis=0)
    sum_all = np.sum(sum_class_true)
    acc = np.sum(np.diagonal(mat / sum_all))
    recall = np.diagonal(mat / sum_class_true.reshape(sum_class_true.shape[0], 1))
    precision = np.diagonal(mat / sum_class_pred.reshape(1, sum_class_true.shape[0]))
    f_score = 2 / (1 / recall + 1 / precision)
    freq = sum_class_true / sum_all
    mat = mat
    return mat, recall, precision, f_score, acc, q8, freq


def reduceToQ10(mat13, decoder):
    q13 = ['m', 'q', 'p', 'a', 'z', 'l', 'b', 'e', 'g', 'i', 'h', 's', 't', '-']
    q10 = {'0': 0, 'h': 1, 'c': 2, 'a': 3, 't': 4, 'p': 5, 's': 6, 'g': 7, 'e': 8, 'm': 9, 'i': 10, 'b': 11, '-': 12}
    a_group = ['a', 'z']
    p_group = ['q', 'p']
    a_inx = [decoder.word_index[w] for w in a_group]
    rdi = {value: key for key, value in decoder.word_index.items()}
    rdi[0] = '0'
    mat10 = np.zeros([13, 13], dtype=int)
    for i in range(mat13.shape[0]):
        for j in range(mat13.shape[1]):
            mi_letter = ''
            if rdi[i] in a_group:
                mi_letter = 'a'  # letter of i
            elif rdi[i] in p_group:
                mi_letter = 'p'
            else:
                mi_letter = rdi[i]
            mj_letter = ''
            if rdi[j] in a_group:
                mj_letter = 'a'  # letter of i
            elif rdi[j] in p_group:
                mj_letter = 'p'
            else:
                mj_letter = rdi[j]
            mat10[q10[mi_letter], q10[mj_letter]] += mat13[i, j]
    mat = mat10
    sum_class_true = np.sum(mat, axis=-1)
    sum_class_pred = np.sum(mat, axis=0)
    sum_all = np.sum(sum_class_true)
    acc = np.sum(np.diagonal(mat / sum_all))
    recall = np.diagonal(mat / sum_class_true.reshape(sum_class_true.shape[0], 1))
    precision = np.diagonal(mat / sum_class_pred.reshape(1, sum_class_true.shape[0]))
    f_score = 2 / (1 / recall + 1 / precision)
    freq = sum_class_true / sum_all
    mat = mat
    return mat, recall, precision, f_score, acc, q10, freq


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
    recall = np.diagonal(mat / sum_class_true.reshape(sum_class_true.shape[0], 1))
    precision = np.diagonal(mat / sum_class_pred.reshape(1, sum_class_true.shape[0]))
    freq = sum_class_true / sum_all
    f_score = 2 / (1 / recall + 1 / precision)
    mat = mat
    m = mcc(ym, y_m)
    return mat, recall, precision, f_score, acc, freq, m


def print_performance(out, mat_pred, mat_true, space_inx, i, decoder):
    # if i == 8:
    #     mat, recall, precision, f_score, acc, q_dict, freq = reduceToQ8(mat, decoder)
    # if i == 10:
    #     mat, recall, precision, f_score, acc, q_dict, freq = reduceToQ10(mat, decoder)
    ex_real = ex_accuracy_real(mat_true, mat_pred, space_inx, True)
    mat, recall, precision, f_score, acc, freq, m = confusion_matrix(y_true=mat_true, y_pred=mat_pred,
                                                                     decoder=decoder)
    q_dict = {1: 'h', 2: 'c', 3: 't', 4: 'z', 5: '-', 6: 's', 7: 'a', 8: 'g', 9: 'q', 10: 'p', 11: 'e', 12: 'b',
              13: 'm', 14: 'i'}

    with open(out + "/performance" + str(i) + ".csv", "w") as f:
        f.write(str(acc) + "\n")
        f.write(str(mat) + "\n")
        f.write("Class dict: " + str(q_dict) + "\n")

        np.set_printoptions(formatter={'float_kind': '{:0.4f}'.format})
        np.set_printoptions(precision=4, suppress=True)
        f.write("*********\n")
        f.write("  | Recall  |  Precision  | F_score  | Freq \n")
        for c in q_dict.keys():
            if (c == '-'):
                continue
            f.write("%s | %1.5f |   %1.5f   | %1.5f  | %1.5f \n" % (
                c, recall[q_dict[c]], precision[q_dict[c]], f_score[q_dict[c]], freq[q_dict[c]]))


def gen_out_csv(y_test_true, y_test_pred, space_inx, test_df, X_test, test_seqs, revsere_decoder_index, out):
    decoded_y_true = []
    decoded_y_pred = []
    pertarget_performance = []
    ptp = ex_accuracy_real(y_test_true, y_test_pred, space_inx, by_target=True)
    for i in range(len(X_test)):
        decoded_y_true.append(decode_results(test_seqs[i], y_test_true[i], revsere_decoder_index))
        decoded_y_pred.append(decode_results(test_seqs[i], y_test_pred[i], revsere_decoder_index))
        pertarget_performance.append(ptp[i])

    out_df = pd.DataFrame()
    out_df["id"] = test_df.id.values
    out_df["pid"] = test_df.pid.values
    out_df["len"] = test_df.len.values
    out_df["input"] = test_df.input.values
    out_df["prediction"] = decoded_y_pred
    out_df["true"] = decoded_y_true
    out_df["performance"] = pertarget_performance

    with open(out + "/out.csv", "w") as f:
        out_df.to_csv(f, index=False)


def get_reorder_vec(model_num, original_order_dict, path, q_num, e_num, w_num, d_num):
    letters_dict = get_order_of_letters(model_num, path, q_num, e_num, w_num, d_num)
    position_dict = {v: k for k, v in letters_dict.items()}
    reorder_dict = {k: original_order_dict[position_dict[k]] for k in range(1, len(letters_dict) + 1)}
    return [reorder_dict[i] for i in range(1, len(letters_dict) + 1)]


def get_order_of_letters(model_num, path, q_num, e_num, w_num, d_num):
    decoder_file = open(path + ' /model_nr' + str(q_num) + '_e' + str(e_num) + '_m' + str(model_num) + '_w' + str(
        w_num) + '_d' + str(d_num) + '/decoder.pickle', 'rb')
    decoder = pickle.load(decoder_file)
    return decoder.word_index


def sort_matrices_by_letters(amount_of_models, models_matrices, path, q_num, e_num, w_num, d_num):
    # sort all letters by the order of model 1.

    num_of_matrices = len(models_matrices[1])
    order_by = get_order_of_letters(1, path, q_num, e_num, w_num, d_num)

    for m in range(1, amount_of_models + 1):
        for i in range(num_of_matrices):
            reoredr_vec = get_reorder_vec(m, order_by, path, q_num, e_num, w_num, d_num)
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


def load_test_pred(idx, path, q_num, e_num, w_num, d_num):
    string_to_load = path + '/model_nr' + str(q_num) + '_e' + str(e_num) + '_m' + str(idx) + '_w' + str(
        w_num) + '_d' + str(d_num) + '/out_test_pred.npy'
    return np.load(string_to_load)


def load_name_vec(idx, path, q_num, e_num, w_num, d_num):
    alttype = np.dtype([('f0', 'U8'), ('f1', 'U8'), ('f2', 'U8'), ('f3', 'U8')])
    string_to_load = path + '/model_nr' + str(q_num) + '_e' + str(e_num) + '_m' + str(idx) + '_w' + str(
        w_num) + '_d' + str(d_num) + '/out.csv'
    return np.genfromtxt(string_to_load, delimiter=',', dtype=alttype, names=True)['pid']


def avg_secondary_struct(models_matrices, amount):
    num_of_matrices = len(models_matrices[1])
    num_of_letters = len(models_matrices[1][0])
    num_of_proj = len(models_matrices[1][0][0])

    avg_mat = np.zeros((num_of_matrices, num_of_letters, num_of_proj))

    # for each matrix out of 1598 max matrices (which are proteins)
    for i in range(num_of_matrices):
        # for each letter in the amino chain of the protein out of 700 max
        for j in range(num_of_letters):
            # for each probability out of 15 max secondary probabilities
            for k in range(num_of_proj):
                sum_of_current_secondary_proj = 0
                # sum the projection for the secondary structure for each matrix
                for m in range(1, amount + 1):
                    sum_of_current_secondary_proj += models_matrices[m][i][j][k]
                avg_mat[i][j][k] = sum_of_current_secondary_proj / amount

    return avg_mat


def avg_secondary_struct2(models_matrices, amount, weights):
    num_of_matrices = len(models_matrices[1])
    num_of_letters = len(models_matrices[1][0])
    num_of_proj = len(models_matrices[1][0][0])

    avg_mat = np.zeros((num_of_matrices, num_of_letters, num_of_proj))

    for i in range(1, amount + 1):
        avg_mat += models_matrices[i] * weights[i]

    return avg_mat / amount


def preform_avg(path, q_num, e_num, w_num, d_num, amount):
    # create matrices
    weights = [0, 1, 1, 1, 1, 1, 1]
    matrices = [None]
    name_vecs = [None]
    models_matrices = [None]
    for i in range(1, amount + 1):
        matrices.append(load_test_pred(i, path, q_num, e_num, w_num, d_num))
        name_vecs.append(load_name_vec(i, path, q_num, e_num, w_num, d_num))
        models_matrices.append(sort_pred_matrix(mat=matrices[i], name_vec=name_vecs[i]))

    # models_matrices = np.array(models_matrices)

    # in this point, all matrices are sorted by the pid from the out file.
    # now we need (?) TODO sort by the letters of the scondary structure
    #  models_matrices = sort_matrices_by_letters(amount, models_matrices)
    # avg1 = avg_secgondary_struct(models_matrices, amount)
    avg2 = avg_secondary_struct2(models_matrices, amount, weights)
    # print(np.equal(avg1, avg2))
    return avg2


def output_data(path, q_num, e_num, w_num, d_num):
    out = path + '/averages_model'
    avg_mat = np.load(path + '/averages_model/avg_pred.npy')
    true_mat = np.load(path + '/model_nr' + str(q_num) + '_e' + str(e_num) + '_m1_w' + str(w_num) + '_d' + str(
        d_num) + '/out_test_true.npy')
    alttype = np.dtype([('f0', 'U8'), ('f1', 'U8'), ('f2', 'U8'), ('f3', 'U8')])
    string_to_load = path + '/model_nr' + str(q_num) + '_e' + str(e_num) + '_m1_w' + str(w_num) + '_d' + str(
        d_num) + '/out.csv'
    name_vec = np.genfromtxt(string_to_load, delimiter=',', dtype=alttype, names=True)['pid']
    true_mat = sort_pred_matrix(mat=true_mat, name_vec=name_vec)
    decoder_file = open(path + '/model_nr' + str(q_num) + '_e' + str(e_num) + '_m1_w' + str(w_num) + '_d' + str(
        d_num) + '/decoder.pickle', 'rb')
    decoder1 = pickle.load(decoder_file)
    # performance
    # ptp = ex_accuracy_real(y_test_true, y_test_pred, space_inx, by_target=True)
    space_inx = decoder1.word_index['-']
    print_performance(out=out, mat_pred=avg_mat, mat_true=true_mat, i=q_num, space_inx=space_inx, decoder=decoder1)
    # out.csv

    # numpy out_test_true

    # decoder.pickle

    # encoder.pickle

    # model.pickle

    # model_e150 JSON

    # trajectory.dat

    # model_e150.h5

    # given the averaged matrices, now we will produce the files requested.
    # as a way of regularity, all data is set by the order of model 1.


def testDecoders(path, q_num, e_num, w_num, d_num):
    decoder_file1 = open(
        path + '/model_nr' + str(q_num) + '_e' + str(e_num) + '_m' + str(1) + '_w' + str(w_num) + '_d' + str(
            d_num) + ' /decoder.pickle', 'rb')
    decoder1 = pickle.load(decoder_file1)

    decoder_file2 = open(
        path + '/model_nr' + str(q_num) + '_e' + str(e_num) + '_m' + str(2) + '_w' + str(w_num) + '_d' + str(
            d_num) + ' /decoder.pickle', 'rb')
    decoder2 = pickle.load(decoder_file2)

    decoder_file3 = open(
        path + '/model_nr' + str(q_num) + '_e' + str(e_num) + '_m' + str(3) + '_w' + str(w_num) + '_d' + str(
            d_num) + ' /decoder.pickle', 'rb')
    decoder3 = pickle.load(decoder_file3)

    print("hello")


def store_data(path, q_num, e_num, w_num, d_num, amount_of_models):
    avg_mat = preform_avg(path, q_num, e_num, w_num, d_num, amount_of_models)
    out = path + '/averages_model'

    # performance

    # numpy out test pred
    np.save(out + "/avg_pred.npy", avg_mat, allow_pickle=True)


def tests():
    ofer = np.array([[[11, 12, 13], [14, 15, 15], [16, 17, 18], [19, 20, 21]],
                     [[21, 22, 23], [24, 25, 26], [27, 28, 29], [30, 31, 32]]])
    john = np.array([[[10, 2, 1], [18, 1, 5], [1, 1, 8], [9, 2, 2]],
                     [[4441, 12, 73], [29, 28, 26], [25, 28, 49], [300, 3, 2]]])

    ringo = np.array([[[0, 7, 1], [18, 41, 577], [14, 1, 8], [94, 82, 92]],
                      [[41, 172, 773], [297, 285, 26], [254, 258, 449], [3004, 3, 2]]])

    matrices = np.array([None, ofer, john, ringo])
    avg1 = avg_secondary_struct(matrices, 3)
    #     print(avg1)
    avg2 = avg_secondary_struct2(matrices, 3, [0, 1, 1, 1])
    #     print(avg2)
    print(np.array_equal(avg1, avg2))
    # example to understand the order letter
    # ofer = np.array([[["a", 'b', "c"], ['d', 'e', 'f'], ['g', 'h', 'i'], ['j', 'k', 'l']],
    #                      [['m', 'n', 'o'], ['p', 'q', 'r'], ['s', 't', 'u'], ['v', 'w', 'x']]])
    #     ofer_name_dict = {"ran": 0, "jhon": 1, "ben": 2}
    #     order_dict = {"jhon": 0, "ben": 1, "ran": 2}
    #
    #     # reorder_vec = np.array([2, 0, 1])
    #
    #     position_dict = {v: k for k, v in ofer_name_dict.items()}
    #     reorder_dict = {k: order_dict[position_dict[k]] for k in range(len(ofer_name_dict))}
    #     reorder_vec = [reorder_dict[i] for i in range(len(ofer_name_dict))]
    #     ofer[0] = ofer[0][:, reorder_vec]
    #     print(ofer[0])


def main():
    q_num = 13
    e_num = 150
    w_num = 0
    d_num = 1
    path = 'q' + str(q_num) + '_nr'
    amount_of_models = 6

    store_data(path, q_num, e_num, w_num, d_num, amount_of_models)
    output_data(path, q_num, e_num, w_num, d_num)
    print("done")
    # tests()


if __name__ == '__main__':
    main()

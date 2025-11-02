#Splitting the table into training and test sets by patients ID.

import random
def shuffling(mylist):
    random.seed()
    newList = []
    for i in range(0,len(mylist)):
        newList.append(mylist[i])

    for i in range(0, len(newList)):
        rindex = random.randint(0, len(newList)-1)
        temp = newList[i]
        newList[i] = newList[rindex]
        newList[rindex] = temp

    return newList


def splitData(dataset):

    noRow = dataset.shape[0]
    all_oryg_id_list = []
    oryg_id_list = []


    for i in range(0, noRow):
        long_id = str(dataset.iat[i, 0])
        short_id = int(long_id[:6])
        all_oryg_id_list.append(short_id)
        if short_id not in oryg_id_list:
            oryg_id_list.append(short_id)

    id_list = shuffling(oryg_id_list)


    no_in_train = int(len(id_list)*0.5) #50% + 50%

    train_id_list = []
    test_id_list = []

    for i in range(0,len(id_list)):
        if i<=no_in_train:
            train_id_list.append(id_list[i])
        else:
            test_id_list.append(id_list[i])

    #=============================================

    train_obj_index_list = []
    test_obj_index_list = []


    for i in range(0,len(all_oryg_id_list)):
        short_id = all_oryg_id_list[i]
        if short_id in train_id_list:
            train_obj_index_list.append(i)
        else:
            if short_id in test_id_list:
                test_obj_index_list.append(i)
            else:
                print("Error",id); exit()


    train_table = dataset.iloc[train_obj_index_list, :]
    test_table = dataset.iloc[test_obj_index_list, :]

    return train_table,test_table



if __name__ == "__main__":

    print("STOP")



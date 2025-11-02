#Extracting the columns needed to determine decisions (behavioral patterns)

SEL_ATTR_LIST = ["MEAN_BREATH_next", "MIN_BREATH_next","MAX_BREATH_next","MEAN_BREATH_now", "MIN_BREATH_now","MAX_BREATH_now"]

def getSubTable(dataset):

    kolumny = list(dataset.columns.values)

    column_index_list = []
    for i in range(0, len(SEL_ATTR_LIST)):
        attr_name = SEL_ATTR_LIST[i]
        JEST = False
        for j in range(0, len(kolumny)):
            if attr_name==kolumny[j]:
                column_index_list.append(j)
                JEST = True
                break
        if not JEST:
            print("ERROR10 No:",attr_name)
            exit()


    sel_features = dataset.iloc[:, column_index_list]

    return sel_features

#----------------

def buidDecisionColumnDataSets(train_data,test_data):

    decision_columns_train_data = getSubTable(train_data)

    decision_columns_test_data = getSubTable(test_data)

    return decision_columns_train_data, decision_columns_test_data

if __name__ == "__main__":
    print("STOP")

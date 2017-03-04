from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, SelectPercentile,VarianceThreshold,f_regression,chi2,f_classif
from sklearn.metrics import f1_score
from sklearn.feature_extraction import DictVectorizer

def average (array):
    sum = 0
    count = 0
    for a in array:
        sum +=a
        count +=1
    return str(sum/count)

import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


def PredictionScoreLeaveOneOut (X,y, limit, columnName):
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.svm import SVC, LinearSVC
    import matplotlib.pyplot as plt

    names = ["Linear SVM","Nearest Neighbors",  "RBF SVM", "Decision Tree",
     "Random Forest", "AdaBoost", "Naive Bayes"]
    # names = ["Linear SVM","Linear SVM","Linear SVM","Linear SVM"]

    classifiers = [
    SVC(kernel="linear", C=0.025, probability=True),
    KNeighborsClassifier(3),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB()]

    outFile = open('output.txt', 'a')

    vec = DictVectorizer()

    for name, clf in zip(names, classifiers):
        try:
            accuracy = 0.0
            count=0.0
            total_accuracy= 0.0
            total_f1= 0.0
            total_precision= 0.0
            total_recall= 0.0

            count = 1.0

            from sklearn.model_selection import LeaveOneOut
            loo = LeaveOneOut()
            loo.get_n_splits(X)

            # print(loo)
            y_test_all = []
            y_pred_all=[]
            accuracy_total = 0
            count = 0
            for train_index, test_index in loo.split(X):
                # print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                from sklearn.feature_extraction.text import CountVectorizer
                count_vect = CountVectorizer()
                X_train_fit = count_vect.fit(X_train)
                X_train_counts = X_train_fit.transform(X_train)
                X_test_counts = X_train_fit.transform(X_test)

                from sklearn.feature_extraction.text import TfidfTransformer
                tfidf_transformer = TfidfTransformer()
                fit = tfidf_transformer.fit(X_train_counts)
                X_train_tfidf = fit.transform(X_train_counts)
                X_test_tfidf = fit.transform(X_test_counts)

                X_train_counts = X_train_tfidf
                X_test_counts = X_test_tfidf
                try:
                    clf.fit(X_train_counts.toarray(), y_train)
                    accuracy_total += clf.score(X_test_counts.toarray(), y_test)
                    count +=1
                    y_pred = clf.predict(X_test_counts.toarray())
                    #
                    # binary_predictions = [x if x == 'good' else 0 for x in y_pred]
                    # binary_predictions = [x if x == 0 else 1 for x in binary_predictions]
                    #
                    # binary_labels = [x if x == 'good' else 0 for x in y_test]
                    # binary_labels = [x if x == 0 else 1 for x in binary_labels]
                    y_pred_all.append(y_pred[0])
                    y_test_all.append(y_test[0])

                except BaseException as b:
                    print (b)


            f1 = f1_score(y_test_all, y_pred_all,average='weighted')
            precision = precision_score(y_test_all, y_pred_all, average='weighted')
            recall = recall_score(y_test_all, y_pred_all, average='weighted')


            print (str(columnName)+"\t"+str(limit) + "\t" + str(name) +"\t"+ str(accuracy_total/count)+"\t"+ str(f1)+"\t"+str(precision)+"\t"+str(recall))
            outFile.write(str(columnName)+"\t"+str(limit) + "\t" + str(name) +"\t"+ str(accuracy_total/count)+"\t"+ str(f1)+"\t"+str(precision)+"\t"+str(recall)+"\n")
            # acc, f1,prc,rec = classify(clf,X_train,X_test,y_train,y_test)
            #
            # total_accuracy +=acc
            # total_f1 += f1
            # total_precision += prc
            # total_recall += rec

        except BaseException as b:
            print (b)
    outFile.close()

def PredictionScoreLeaveOneOutSpecifyClassifier (X,y, limit, columnName, classifierNames, classifiers):
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.svm import SVC, LinearSVC
    import matplotlib.pyplot as plt

    names = classifierNames

    outFile = open('output.txt', 'a')

    vec = DictVectorizer()

    for name, clf in zip(names, classifiers):
        try:
            accuracy = 0.0
            count=0.0
            total_accuracy= 0.0
            total_f1= 0.0
            total_precision= 0.0
            total_recall= 0.0

            count = 1.0

            from sklearn.model_selection import LeaveOneOut
            loo = LeaveOneOut()
            loo.get_n_splits(X)

            # print(loo)
            y_test_all = []
            y_pred_all=[]
            accuracy_total = 0
            count = 0
            for train_index, test_index in loo.split(X):
                # print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                from sklearn.feature_extraction.text import CountVectorizer
                count_vect = CountVectorizer()
                X_train_fit = count_vect.fit(X_train)
                X_train_counts = X_train_fit.transform(X_train)
                X_test_counts = X_train_fit.transform(X_test)
                #
                from sklearn.feature_extraction.text import TfidfTransformer
                tfidf_transformer = TfidfTransformer()
                fit = tfidf_transformer.fit(X_train_counts)
                X_train_tfidf = fit.transform(X_train_counts)
                X_test_tfidf = fit.transform(X_test_counts)

                X_train_counts = X_train_tfidf
                X_test_counts = X_test_tfidf
                try:
                    clf.fit(X_train_counts.toarray(), y_train)
                    accuracy_total += clf.score(X_test_counts.toarray(), y_test)
                    count +=1
                    y_pred = clf.predict(X_test_counts.toarray())
                    #
                    # binary_predictions = [x if x == 'good' else 0 for x in y_pred]
                    # binary_predictions = [x if x == 0 else 1 for x in binary_predictions]
                    #
                    # binary_labels = [x if x == 'good' else 0 for x in y_test]
                    # binary_labels = [x if x == 0 else 1 for x in binary_labels]
                    y_pred_all.append(y_pred[0])
                    y_test_all.append(y_test[0])

                except BaseException as b:
                    print (b)


            f1 = f1_score(y_test_all, y_pred_all,average='weighted')
            precision = precision_score(y_test_all, y_pred_all, average='weighted')
            recall = recall_score(y_test_all, y_pred_all, average='weighted')


            print (str(columnName)+"\t"+str(limit) + "\t" + str(name) +"\t"+ str(accuracy_total/count)+"\t"+ str(f1)+"\t"+str(precision)+"\t"+str(recall))
            outFile.write(str(columnName)+"\t"+str(limit) + "\t" + str(name) +"\t"+ str(accuracy_total/count)+"\t"+ str(f1)+"\t"+str(precision)+"\t"+str(recall)+"\n")
            # acc, f1,prc,rec = classify(clf,X_train,X_test,y_train,y_test)
            #
            # total_accuracy +=acc
            # total_f1 += f1
            # total_precision += prc
            # total_recall += rec

        except BaseException as b:
            print (b)
    outFile.close()

def k_fold_generator(X, y, k_fold):
    subset_size = int(len(X) / k_fold)
    for k in range(k_fold):
        X_train = X[:k * subset_size] + X[(k + 1) * subset_size:]
        X_valid = X[k * subset_size:][:subset_size]
        y_train = y[:k * subset_size] + y[(k + 1) * subset_size:]
        y_valid = y[k * subset_size:][:subset_size]

        yield X_train, y_train, X_valid, y_valid

def classifytop20_duplicatewords (scoreTableName,columnName):
    # print (scoreTableName +columnName)
    import pymysql
    conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='62217769', db='crawler')
    cur = conn.cursor()
    cur.execute("SELECT id,datasetName,endpointUrl,subjectid FROM endpoints WHERE subject is not null and subjectid !=5 and subjectid !=7 ")

    endpointContentsDict = {}
    targetDict = {}

    for row in cur:
            # print(str(row[0])+"-"+str(row[2]))
        cur2 = conn.cursor()
        colNum=0
        try:
            cur2.execute("SELECT "+columnName+",endpointid,tf,stf FROM "+scoreTableName+" where endpointid=" + str(row[0]))
        except BaseException as b :
            cur2.execute(
                "SELECT " + columnName + ",endpointid,count FROM " + scoreTableName + " where endpointid=" + str(
                    row[0]))
            colNum  = 2

        all1 = ""
        count = 0
        for row2 in cur2:
            if colNum!= 2:
                if columnName == "word":
                    colNum=2
                else:
                    colNum=3

            for i in range(1,int(row2[colNum])):
                all1 += " " + str(row2[0])

            count+=1
        if all1!='':
            # print(count)
            if row[0] in endpointContentsDict.keys():
                endpointContentsDict[row[0]] += all1
                # target[row[0]] = row[3]
            else:
                endpointContentsDict[row[0]] = all1
                targetDict[row[0]] = row[3]

    endpointContents = []
    target = []
    for c in endpointContentsDict.items():
        endpointContents.append(c[1])
        target.append(targetDict[c[0]])


    import numpy as np
    X = np.array(endpointContents)
    y = np.array(target)

    PredictionScoreLeaveOneOut (X,y,0,"duplicate"+scoreTableName +columnName)

def classifytop20(scoreTableName, columnName):
    # print (scoreTableName +columnName)
    import pymysql
    conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='62217769', db='crawler')
    cur = conn.cursor()
    cur.execute(
        "SELECT id,datasetName,endpointUrl,subjectid FROM endpoints WHERE subject is not null and subjectid !=5 and subjectid !=7 ")

    endpointContentsDict = {}
    targetDict = {}

    for row in cur:
        # print(str(row[0])+"-"+str(row[2]))
        cur2 = conn.cursor()
        cur2.execute(
            "SELECT " + columnName + ",endpointid FROM " + scoreTableName + " where endpointid=" + str(row[0]))
        all1 = ""
        count = 0
        for row2 in cur2:
            all1 += " " + str(row2[0])

            count += 1
        if all1 != '':
            # print(count)
            if row[0] in endpointContentsDict.keys():
                endpointContentsDict[row[0]] += all1
                # target[row[0]] = row[3]
            else:
                endpointContentsDict[row[0]] = all1
                targetDict[row[0]] = row[3]

    endpointContents = []
    target = []
    for c in endpointContentsDict.items():
        endpointContents.append(c[1])
        target.append(targetDict[c[0]])

    import numpy as np
    X = np.array(endpointContents)
    y = np.array(target)

    PredictionScoreLeaveOneOut(X, y, 0, scoreTableName + columnName)

    # print(X_train, X_test, y_train, y_test)

# from sklearn import svm
# from sklearn.model_selection import cross_val_score
#
#
# PredictionScoreLeaveOneOut(X_train=X_train_tfidf, X_test=[], y_train=y, y_test=[], cross_validation=True, k_fold=10)
#
# clf = svm.SVC(kernel='linear', C=1)
# scores = cross_val_score(clf, X_train_tfidf, y, cv=10)
#
# print (average(scores))
#
# from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB()
# scores = cross_val_score(clf, X_train_tfidf, y, cv=10)
#
# print (average(scores))
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# scores = cross_val_score(clf, X_train_tfidf, y, cv=5)
#
# print (scores)

# kf = KFold(n_splits=3)
# totalscore = 0
# count = 0
# for train, test in kf.split(X):
#     X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
#     from sklearn.feature_extraction.text import CountVectorizer
#     count_vect = CountVectorizer()
#     X_train_fit = count_vect.fit(X_train)
#     X_train_counts = X_train_fit.transform(X_train)
#     X_test_counts = X_train_fit.transform(X_test)
#
#     from sklearn.feature_extraction.text import TfidfTransformer
#     tfidf_transformer = TfidfTransformer()
#     X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#     X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
#
#     from sklearn.naive_bayes import MultinomialNB
#     clf = MultinomialNB().fit(X_train_tfidf, y_train)
#     score = clf.score(X_test_tfidf,y_test)
#     totalscore +=score
#     count += 1
#     # print (str(totalscore/count))
#     print (str(score))
def classifyHigherToLower (scoreTableName,columnName, sortColumnName):
    # print (scoreTableName +columnName)
    max = 100
    for limit in range(1, max):
        import pymysql
        conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='62217769', db='crawler')
        cur = conn.cursor()
        cur.execute("SELECT id,datasetName,endpointUrl,subjectid FROM endpoints WHERE subject is not null and subjectid !=5 and subjectid !=7 ")

        endpointContentsDict = {}
        targetDict = {}


        for row in cur:
                # print(str(row[0])+"-"+str(row[2]))
            cur2 = conn.cursor()
            cur2.execute(
                "SELECT "+columnName+",endpointid FROM "+scoreTableName+" where endpointid=" + str(row[0])+" order by "+sortColumnName+" desc LIMIT " +str(limit))
#            cur2.execute(
#                "SELECT "+columnName+",endpointid FROM "+scoreTableName+" where endpointid=" + str(row[0])+" order by "+sortColumnName+" desc LIMIT " +str(limit))
            all1 = ""
            count = 0
            for row2 in cur2:
                all1 += " " + str(row2[0])

                count+=1
            if all1!='':
                # print(count)
                if row[0] in endpointContentsDict.keys():
                    endpointContentsDict[row[0]] += all1
                    # target[row[0]] = row[3]
                else:
                    endpointContentsDict[row[0]] = all1
                    targetDict[row[0]] = row[3]

        endpointContents = []
        target = []
        for c in endpointContentsDict.items():
            endpointContents.append(c[1])
            target.append(targetDict[c[0]])


        import numpy as np
        X = np.array(endpointContents)
        y = np.array(target)


        PredictionScoreLeaveOneOut (X,y, limit, scoreTableName+"_"+columnName+"_"+sortColumnName+"_Limit")

def classifyWindow (scoreTableName,columnName, sortColumnName, windowSize):
    # print (scoreTableName +columnName)
    max = 100
    for limit in range(0, max):
        import pymysql
        conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='62217769', db='crawler')
        cur = conn.cursor()
        cur.execute("SELECT id,datasetName,endpointUrl,subjectid FROM endpoints WHERE subject is not null and subjectid !=5 and subjectid !=7 ")

        endpointContentsDict = {}
        targetDict = {}


        for row in cur:
                # print(str(row[0])+"-"+str(row[2]))
            cur2 = conn.cursor()
            cur2.execute(
                "SELECT "+columnName+",endpointid FROM "+scoreTableName+" where endpointid=" + str(row[0])+" order by "+sortColumnName+" desc LIMIT " +str(limit)+", "+str(windowSize))
#            cur2.execute(
#                "SELECT "+columnName+",endpointid FROM "+scoreTableName+" where endpointid=" + str(row[0])+" order by "+sortColumnName+" desc LIMIT " +str(limit))
            all1 = ""
            count = 0
            for row2 in cur2:
                all1 += " " + str(row2[0])

                count+=1
            if all1!='':
                # print(count)
                if row[0] in endpointContentsDict.keys():
                    endpointContentsDict[row[0]] += all1
                    # target[row[0]] = row[3]
                else:
                    endpointContentsDict[row[0]] = all1
                    targetDict[row[0]] = row[3]

        endpointContents = []
        target = []
        for c in endpointContentsDict.items():
            endpointContents.append(c[1])
            target.append(targetDict[c[0]])

        import numpy as np
        X = np.array(endpointContents)
        y = np.array(target)

        PredictionScoreLeaveOneOut (X,y, limit, scoreTableName+"_"+columnName+"_"+sortColumnName+"_Offset_"+str(limit)+"_WindowSize_"+str(windowSize))
def classifyAll (scoreTableName,columnName):
    # print (scoreTableName +columnName)
    max = 100001
    for limit in range(100000, max):
        import pymysql
        conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='62217769', db='crawler')
        cur = conn.cursor()
        cur.execute("SELECT id,datasetName,endpointUrl,subjectid FROM endpoints WHERE subject is not null and subjectid !=5 and subjectid !=7 ")

        endpointContentsDict = {}
        targetDict = {}


        for row in cur:
                # print(str(row[0])+"-"+str(row[2]))
            cur2 = conn.cursor()
            cur2.execute(
                "SELECT "+columnName+",endpointid FROM "+scoreTableName+" where endpointid=" + str(row[0])+" LIMIT " +str(limit))
#            cur2.execute(
#                "SELECT "+columnName+",endpointid FROM "+scoreTableName+" where endpointid=" + str(row[0])+" order by "+sortColumnName+" desc LIMIT " +str(limit))
            all1 = ""
            count = 0
            for row2 in cur2:
                all1 += " " + str(row2[0])

                count+=1
            if all1!='':
                # print(count)
                if row[0] in endpointContentsDict.keys():
                    endpointContentsDict[row[0]] += all1
                    # target[row[0]] = row[3]
                else:
                    endpointContentsDict[row[0]] = all1
                    targetDict[row[0]] = row[3]

        endpointContents = []
        target = []
        for c in endpointContentsDict.items():
            endpointContents.append(c[1])
            target.append(targetDict[c[0]])

        import numpy as np
        X = np.array(endpointContents)
        y = np.array(target)

        names = ["Naive Bayes"]
        # names = ["Linear SVM","Linear SVM","Linear SVM","Linear SVM"]

        classifiers = [
            GaussianNB()]

        names = ["Linear SVM", "Nearest Neighbors", "RBF SVM", "Decision Tree",
                 "Random Forest", "AdaBoost", "Naive Bayes"]

        classifiers = [
            SVC(kernel="linear", C=0.025, probability=True),
            KNeighborsClassifier(3),
            SVC(gamma=2, C=1),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            AdaBoostClassifier(),
            GaussianNB()]
        PredictionScoreLeaveOneOutSpecifyClassifier (X,y, limit, scoreTableName+"_"+columnName+"_ClassifyAllLimit_"+str(limit),names,classifiers)

def classifyLowerToHigher (scoreTableName,columnName, sortColumnName):
    # print (scoreTableName +columnName)
    max = 100
    for limit in range(0, max):
        import pymysql
        conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='62217769', db='crawler')
        cur = conn.cursor()
        cur.execute("SELECT id,datasetName,endpointUrl,subjectid FROM endpoints WHERE subject is not null and subjectid !=5 and subjectid !=7 ")

        endpointContentsDict = {}
        targetDict = {}


        for row in cur:
                # print(str(row[0])+"-"+str(row[2]))
            cur2 = conn.cursor()
            cur2.execute(
                "SELECT "+columnName+",endpointid FROM "+scoreTableName+" where endpointid=" + str(row[0])+" order by "+sortColumnName+" desc LIMIT " +str(max-limit)+","+str(limit)+"")
#            cur2.execute(
#                "SELECT "+columnName+",endpointid FROM "+scoreTableName+" where endpointid=" + str(row[0])+" order by "+sortColumnName+" desc LIMIT " +str(limit))
            all1 = ""
            count = 0
            for row2 in cur2:
                all1 += " " + str(row2[0])

                count+=1
            if all1!='':
                # print(count)
                if row[0] in endpointContentsDict.keys():
                    endpointContentsDict[row[0]] += all1
                    # target[row[0]] = row[3]
                else:
                    endpointContentsDict[row[0]] = all1
                    targetDict[row[0]] = row[3]

        endpointContents = []
        target = []
        for c in endpointContentsDict.items():
            endpointContents.append(c[1])
            target.append(targetDict[c[0]])


        import numpy as np
        X = np.array(endpointContents)
        y = np.array(target)


        PredictionScoreLeaveOneOut (X,y, limit, scoreTableName+"_"+columnName+"max"+str(max))

        # print(X_train, X_test, y_train, y_test)

    # from sklearn import svm
    # from sklearn.model_selection import cross_val_score
    #
    #
    # PredictionScoreLeaveOneOut(X_train=X_train_tfidf, X_test=[], y_train=y, y_test=[], cross_validation=True, k_fold=10)
    #
    # clf = svm.SVC(kernel='linear', C=1)
    # scores = cross_val_score(clf, X_train_tfidf, y, cv=10)
    #
    # print (average(scores))
    #
    # from sklearn.naive_bayes import MultinomialNB
    # clf = MultinomialNB()
    # scores = cross_val_score(clf, X_train_tfidf, y, cv=10)
    #
    # print (average(scores))
    # from sklearn.naive_bayes import GaussianNB
    # clf = GaussianNB()
    # scores = cross_val_score(clf, X_train_tfidf, y, cv=5)
    #
    # print (scores)

    # kf = KFold(n_splits=3)
    # totalscore = 0
    # count = 0
    # for train, test in kf.split(X):
    #     X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    #     from sklearn.feature_extraction.text import CountVectorizer
    #     count_vect = CountVectorizer()
    #     X_train_fit = count_vect.fit(X_train)
    #     X_train_counts = X_train_fit.transform(X_train)
    #     X_test_counts = X_train_fit.transform(X_test)
    #
    #     from sklearn.feature_extraction.text import TfidfTransformer
    #     tfidf_transformer = TfidfTransformer()
    #     X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    #     X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
    #
    #     from sklearn.naive_bayes import MultinomialNB
    #     clf = MultinomialNB().fit(X_train_tfidf, y_train)
    #     score = clf.score(X_test_tfidf,y_test)
    #     totalscore +=score
    #     count += 1
    #     # print (str(totalscore/count))
    #     print (str(score))


if __name__ == "__main__":
    import os
    from SPARQLWrapper import SPARQLWrapper, JSON,RDF, POST, GET, SELECT, CONSTRUCT, ASK, DESCRIBE
    import time

    categories = ['government',
                  'publications',
                  'geographic',
                  'crossdomain',
                  'media',
                  'schemata',
                  'lifesciences',
                  'usergeneratedcontent']


    outFile = open('output.txt', 'a')
    outFile.write("-----------------\n")

    classifyAll("recommender_comment_wordnet_topic_anlys","term")
    classifyAll("recommender_label_wordnet_topic_anlys","term")
    classifyAll("recommender_comment_wordnet_hypernym_anlys","term")
    classifyAll("recommender_label_wordnet_hypernym_anlys", "term")


    classifyAll("recommender_comment_wordnet_topic_anlys","word")
    classifyAll("recommender_label_wordnet_topic_anlys","word")
    classifyAll("recommender_comment_wordnet_hypernym_anlys","word")
    classifyAll("recommender_label_wordnet_hypernym_anlys", "word")


    classifyWindow("recommender_comment_wordnet_hypernym_anlys_scr_lvl","term", "stfsidfscore",10)

    classifyWindow("recommender_comment_wordnet_hypernym_anlys_scr_lvl","term", "tfidfscore",10)

    classifyWindow("recommender_comment_wordnet_topic_anlys_scr_lvl","term", "tfidfscore",10)

    classifyWindow("recommender_comment_wordnet_topic_anlys_scr_lvl","term", "stfsidfscore",10)

    classifyWindow("recommender_label_wordnet_hypernym_anlys_scr_lvl","term", "stfsidfscore",10)

    classifyWindow("recommender_label_wordnet_hypernym_anlys_scr_lvl","term", "tfidfscore",10)

    classifyWindow("recommender_label_wordnet_topic_anlys_scr_lvl","term", "tfidfscore",10)
    #
    classifyWindow("recommender_label_wordnet_topic_anlys_scr_lvl","term", "stfsidfscore",10)


    classifyWindow("recommender_comment_wordnet_hypernym_anlys_scr","term", "stfsidfscore",10)

    classifyWindow("recommender_comment_wordnet_hypernym_anlys_scr","term", "tfidfscore",10)

    classifyWindow("recommender_comment_wordnet_topic_anlys_scr","term", "tfidfscore",10)

    classifyWindow("recommender_comment_wordnet_topic_anlys_scr","term", "stfsidfscore",10)

    classifyWindow("recommender_label_wordnet_hypernym_anlys_scr","term", "stfsidfscore",10)

    classifyWindow("recommender_label_wordnet_hypernym_anlys_scr","term", "tfidfscore",10)

    classifyWindow("recommender_label_wordnet_topic_anlys_scr","term", "tfidfscore",10)
    #
    classifyWindow("recommender_label_wordnet_topic_anlys_scr","term", "stfsidfscore",10)


    classifyWindow("recommender_comment_wordnet_hypernym_anlys_scr_lvl","term", "stfsidfscore",20)

    classifyWindow("recommender_comment_wordnet_hypernym_anlys_scr_lvl","term", "tfidfscore",20)

    classifyWindow("recommender_comment_wordnet_topic_anlys_scr_lvl","term", "tfidfscore",20)

    classifyWindow("recommender_comment_wordnet_topic_anlys_scr_lvl","term", "stfsidfscore",20)

    classifyWindow("recommender_label_wordnet_hypernym_anlys_scr_lvl","term", "stfsidfscore",20)

    classifyWindow("recommender_label_wordnet_hypernym_anlys_scr_lvl","term", "tfidfscore",20)

    classifyWindow("recommender_label_wordnet_topic_anlys_scr_lvl","term", "tfidfscore",20)
    #
    classifyWindow("recommender_label_wordnet_topic_anlys_scr_lvl","term", "stfsidfscore",20)


    classifyWindow("recommender_comment_wordnet_hypernym_anlys_scr","term", "stfsidfscore",20)

    classifyWindow("recommender_comment_wordnet_hypernym_anlys_scr","term", "tfidfscore",20)

    classifyWindow("recommender_comment_wordnet_topic_anlys_scr","term", "tfidfscore",20)

    classifyWindow("recommender_comment_wordnet_topic_anlys_scr","term", "stfsidfscore",20)

    classifyWindow("recommender_label_wordnet_hypernym_anlys_scr","term", "stfsidfscore",20)

    classifyWindow("recommender_label_wordnet_hypernym_anlys_scr","term", "tfidfscore",20)

    classifyWindow("recommender_label_wordnet_topic_anlys_scr","term", "tfidfscore",20)
    #
    classifyWindow("recommender_label_wordnet_topic_anlys_scr","term", "stfsidfscore",20)
    # classifyHigherToLower("recommender_comment_wordnet_hypernym_anlys_scr_lvl","term", "stfsidfscore")
    #
    # classifyHigherToLower("recommender_comment_wordnet_hypernym_anlys_scr_lvl","term", "tfidfscore")
    #
    # classifyHigherToLower("recommender_comment_wordnet_topic_anlys_scr_lvl","term", "tfidfscore")
    #
    # classifyHigherToLower("recommender_comment_wordnet_topic_anlys_scr_lvl","term", "stfsidfscore")
    #
    # classifyHigherToLower("recommender_label_wordnet_hypernym_anlys_scr_lvl","term", "stfsidfscore")
    #
    # classifyHigherToLower("recommender_label_wordnet_hypernym_anlys_scr_lvl","term", "tfidfscore")
    #
    # classifyHigherToLower("recommender_label_wordnet_topic_anlys_scr_lvl","term", "tfidfscore")

    # classifyHigherToLower("recommender_label_wordnet_topic_anlys_scr_lvl","term", "stfsidfscore")


    # classifyHigherToLower("recommender_comment_tf","word","count")
    #
    # classifyHigherToLower("recommender_label_tf","word","count")
    #
    # classifyHigherToLower("recommender_comment_wordnet_hypernym_anlys_scr","term", "stfsidfscore")
    #
    # classifyHigherToLower("recommender_comment_wordnet_hypernym_anlys_scr","term", "tfidfscore")
    #
    # classifyHigherToLower("recommender_comment_wordnet_topic_anlys_scr","term", "tfidfscore")
    #
    # classifyHigherToLower("recommender_comment_wordnet_topic_anlys_scr","term", "stfsidfscore")
    #
    # classifyHigherToLower("recommender_label_wordnet_hypernym_anlys_scr","term", "stfsidfscore")
    #
    # classifyHigherToLower("recommender_label_wordnet_hypernym_anlys_scr","term", "tfidfscore")
    #
    # classifyHigherToLower("recommender_label_wordnet_topic_anlys_scr","term", "tfidfscore")
    # #
    # classifyHigherToLower("recommender_label_wordnet_topic_anlys_scr","term", "stfsidfscore")


    # classifytop20_duplicatewords("recommender_comment_tf", "word")
    #
    # classifytop20_duplicatewords("recommender_label_tf", "word")
    #
    # classifytop20_duplicatewords("recommender_comment_wordnet_hypernym_anlys", "term")
    #
    # classifytop20_duplicatewords("recommender_comment_wordnet_hypernym_anlys", "word")
    #
    # classifytop20_duplicatewords("recommender_comment_wordnet_topic_anlys", "word")
    #
    # classifytop20_duplicatewords("recommender_comment_wordnet_topic_anlys", "term")
    #
    # classifytop20_duplicatewords("recommender_label_wordnet_hypernym_anlys", "term")
    #
    # classifytop20_duplicatewords("recommender_label_wordnet_hypernym_anlys", "word")
    #
    # classifytop20_duplicatewords("recommender_label_wordnet_topic_anlys", "word")
    # #
    # classifytop20_duplicatewords("recommender_label_wordnet_topic_anlys", "term")

    # classifytop20("recommender_comment_tf", "word")
    #
    # classifytop20("recommender_label_tf", "word")
    #
    # classifytop20("recommender_comment_wordnet_hypernym_anlys_scr_stsi_t20", "term")
    #
    # classifytop20("recommender_comment_wordnet_hypernym_anlys_scr_ti_t20", "word")
    #
    # classifytop20("recommender_comment_wordnet_topic_anlys_scr_ti_t20", "word")
    #
    # classifytop20("recommender_comment_wordnet_topic_anlys_scr_stsi_t20", "term")
    #
    # classifytop20("recommender_label_wordnet_hypernym_anlys_scr_stsi_t20", "term")
    #
    # classifytop20("recommender_label_wordnet_hypernym_anlys_scr_ti_t20", "word")
    #
    # classifytop20("recommender_label_wordnet_topic_anlys_scr_ti_t20", "word")
    # #
    # classifytop20("recommender_label_wordnet_topic_anlys_scr_stsi_t20", "term")


    # classify(21)
    # classify(23)
    # classify(24)
    # classify(25)
    # classify(26)
    # classify(27)

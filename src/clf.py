from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

def evaluate(y_pred, y_score, y_true, label='binary'):
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    return acc, recall, precision, f1, auc_score

def sklearn_logress(x_train, x_test, y_train, y_test):
    
    clf = LogisticRegression().fit(x_train, y_train)
    y_pred_test = clf.predict(x_test)
    y_score_test = clf.decision_function(x_test)            

    acc, recall, precision, f1, auc_score = evaluate(y_pred_test, y_score_test, y_test)
    print('------------------------------------------------------------------------------------')
    print(f"Accuracy:{acc:.3f}\tRecall:{recall:.3f}\t"
          f"Precision: {precision:.3f}\tF1:{f1:.3f}\tAUC:{auc_score:.3f}")

    return dict({"acc":acc, "recall":recall, "precision": precision, "f1":f1, "auc":auc_score})

def pt_evaluate(score, true, task='binary', verbose=True):
    '''
    Evaluate the model with five metrics (for pytorch).
    '''
    pred = (score > 0.5).int()
    acc = tmf.accuracy(pred, true, task=task)
    recall = tmf.recall(pred, true, task=task)
    precision = tmf.precision(pred, true, task =task)
    f1 = tmf.f1_score(pred, true, task=task)
    auc = tmf.auroc(score, true, task=task)

    if verbose:
        print('------------------------------------------------------------------------------------')
        print(f"Accuracy: {acc:.3f}\tRecall: {recall:.3f}\t"
              f"Precision: {precision:.3f}\tF1:{f1:.3f}\tAUC:{auc:.3f}")
    
    return dict({"acc":acc, "recall":recall, "precision": precision, "f1":f1, "auc":auc})
    
    

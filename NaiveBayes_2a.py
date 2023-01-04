from math import sqrt
from math import exp
from math import pi

training = []
test = []
class_dict = {}
class_valuedict={}  #dictionary to store class values and their corresponding integer value
allclasses = []
no_cols = 0


def convertStringClasstoInteger(training):  #converting class value 'W' and 'M' to 0 and 1 respectively
    for row in training:
        allclasses.append(row[no_cols - 1])
    classes = set(allclasses)
    print("\n")
    for i, value in enumerate(classes):
        class_dict[value] = i
        class_valuedict[i]=value
        print('%s is taken as  %d' % (value, i))
    print("\n")
    for row in training:
        row[no_cols - 1] = class_dict[row[no_cols - 1]] #appending 0 and 1 to clas column of training dataset



# call naive baiyes algorithm
def predict(training, test):
    predicted = naive_bayes(training, test)
    return predicted

#grouping data according to class
def group_by_class(training):
    group = {}
    for i in range(len(training)):
        row = training[i]
        classs = row[-1]
        if (classs not in group):
            group[classs] = list()
        group[classs].append(row)
    return group


# find mean of a list of numbers
def mean(values):
    return sum(values) / float(len(values))


# find standard deviation of a list of numbers
def stndevn(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)

#group data for each class and find statistical data
def grouping_by_class(training):
    groups = group_by_class(training)
    stats = {}
    for classes, row in groups.items():
        stats[classes] = find_statistics(row)
    return stats


# find gaussian probability function for each x value
def calculate_probability(x, mean, stdev , class_value):
    expo = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    gaus_prob= (1 / (sqrt(2 * pi) * stdev)) * expo
    print("P(",x,"|",class_valuedict[class_value],") is ",gaus_prob)
    return gaus_prob

# find the mean, stdev and length for each column
def find_statistics(training):
    stats = [(mean(data), stndevn(data), len(data)) for data in zip(*training)]
    del (stats[-1]) #removing length from stats list
    return stats

# find the probabilities of predicting each class for a given row
def findclassprob(groupedclasses, row):
    total_rows = sum([groupedclasses[label][0][2] for label in groupedclasses]) #find total rows in dataset
    probabilities = {}
    print("-----------------------------------------------------------------------------------------------------------------------------------")
    print("\nData: ", row)
    print("\nGaussian Probability parameters:\n")
    for class_value, class_statistics in groupedclasses.items():
        probabilities[class_value] = groupedclasses[class_value][0][2] / float(total_rows) #find class probability
        for i in range(len(class_statistics)):
            mean, stdev, var = class_statistics[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev,class_value) # find the probabilities of predicting each class for a given row
        print("\n")
    return probabilities



# Naive Bayes Algorithm
def naive_bayes(train, test):
    groupedclasses = grouping_by_class(train)
    for row in test:
        classprobabilities = findclassprob(groupedclasses, row)
        max_prob_label = None
        max_prob = -1
        for classes, probability in classprobabilities.items():
            if max_prob_label is None or probability > max_prob: #returning class having maximum parobability
                max_prob_label = classes
                max_prob = probability
        print("Predicted class for ",row,"is ",class_valuedict[max_prob_label],"with probability ",max_prob)

with open("1a_train.txt") as file:
    for line in file:
        line = line.rstrip(",\r\n")
        patt = line.replace('(', '').replace(')', '').replace(' ', '').strip()
        row = list(patt.split(","))
        no_cols = len(row)
        training.append(row)
    for x in range(len(training)):
        for y in range(no_cols - 1):
            training[x][y] = float(training[x][y])
    print("Training Data:")
    print(training)
    convertStringClasstoInteger(training)

with open("1a_test.txt") as fic:
    for line in fic:
        line = line.rstrip(",\r\n")
        patt = line.replace('(', '').replace(')', '').replace(' ', '').strip()
        row = list(patt.split(","))
        no_of_cols = len(row)
        test.append(row)
    for x in range(len(test)):
        for y in range(no_of_cols):
            test[x][y] = float(test[x][y])
    print("Test Data:")
    print(test)


predict(training, test)  #calling prediction function


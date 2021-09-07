import csv
import itertools
import matplotlib.pyplot as plt

def generateFrequentItemSet(CandidateList, noOfTransactions, minimumSupport, dataSet, fatherFrequentArray):
    frequentItemsArray = []
    temp=[]

    for i in range(len(CandidateList)):
            support = (CandidateList[i][1] * 1.0 / noOfTransactions) * 100
            if support >= minimumSupport:
                temp.append(CandidateList[i][0])
                temp.append(CandidateList[i][1])
                frequentItemsArray.append(temp)
            temp=[]
    for k in frequentItemsArray:
        fatherFrequentArray.append(k)

    if len(frequentItemsArray) == 1 or len(frequentItemsArray) == 0:
        return fatherFrequentArray

    else:

        generateCandidateSets(dataSet, frequentItemsArray, noOfTransactions, minimumSupport,fatherFrequentArray)

def generateCandidateSets(dataSet, frequentItemsArray, noOfTransactions, minimumSupport,fatherFrequentArray):
    onlyElements = []
    arrayAfterCombinations = []
    candidateSetArray = []
    for i in range(len(frequentItemsArray)):
        onlyElements.append(frequentItemsArray[i][0])

    for item in onlyElements:
        tempCombinationArray = []
        k = onlyElements.index(item)
        for i in range(k + 1, len(onlyElements)):
            for j in item:
                if j not in tempCombinationArray:
                    tempCombinationArray.append(j)
            for m in onlyElements[i]:
                if m not in tempCombinationArray:
                    tempCombinationArray.append(m)
            arrayAfterCombinations.append(tempCombinationArray)
            tempCombinationArray = []
    sortedCombinationArray = []
    uniqueCombinationArray = []
    for i in arrayAfterCombinations:
        sortedCombinationArray.append(sorted(i))
    for i in sortedCombinationArray:
        if i not in uniqueCombinationArray:
            uniqueCombinationArray.append(i)
    arrayAfterCombinations = uniqueCombinationArray
    temp=[]
    for item in arrayAfterCombinations:
        count = 0
        for transaction in dataSet:
            if set(item).issubset(set(transaction)):
                count = count + 1
        if count != 0:
            temp.append(item)
            temp.append(count)
            candidateSetArray.append(temp)
        temp=[]
    generateFrequentItemSet(candidateSetArray, noOfTransactions, minimumSupport, dataSet, fatherFrequentArray)


def generateAssociationRule(freqSet):
    associationRule = []
    for item in freqSet:
        if isinstance(item[0], list):
            if len(item[0]) != 0:
                length = len(item[0]) - 1
                while length > 0:
                    combinations = list(itertools.combinations(item[0], length))
                    temp = []
                    LHS = []
                    for RHS in combinations:

                        LHS = set(item[0]) - set(RHS)
                        temp.append(list(LHS))
                        temp.append(list(RHS))
                        associationRule.append(temp)
                        temp = []
                    length = length - 1
    return associationRule

def aprioriOutput(rules, dataSet, minimumConfidence):
    returnAprioriOutput = []
    for rule in rules:
        supportOfX = 0
        supportOfY = 0
        supportOfXandY = 0
        for transaction in dataSet:
            if set(rule[0]).issubset(set(transaction)):
                supportOfX = supportOfX + 1
            if set(rule[0] + rule[1]).issubset(set(transaction)):
                supportOfXandY = supportOfXandY + 1
            if set(rule[1]).issubset(set(transaction)):
                supportOfY = supportOfY + 1
        supportOfXinPercentage = (supportOfX * 1.0 / noOfTransactions) * 100
        supportOfYinPercentage = (supportOfY * 1.0 / noOfTransactions) * 100
        supportOfXandYinPercentage = (supportOfXandY * 1.0 / noOfTransactions) * 100
        confidence = (supportOfXandYinPercentage / supportOfXinPercentage) * 100
        if confidence >= minimumConfidence:
            LiftAppendString = "Lift: " + str(round((supportOfY*supportOfX)/supportOfXandY))

            returnAprioriOutput.append(rule)

            returnAprioriOutput.append(LiftAppendString)


    return returnAprioriOutput

i=1
itemOfTrans=[]
minSup=2
minConf=10
frequentItemset=[]
with open('groceries.csv', newline='') as csvfile:
    readFile = csv.reader(csvfile, delimiter=',', quotechar='|')
    data = list(readFile) #list trans

    listOfTransactions = {} #dict trans
    for items in data:
        for item in items:
            if item not in listOfTransactions:
               listOfTransactions[item] = 1
            else:
                 listOfTransactions[item] = listOfTransactions[item] + 1

    items=[]
    count=[]
    for key, value in listOfTransactions.items():
        temp = [[key], value]
        itemOfTrans.append(temp)#list transaction with count
        items.append(key)
        count.append(value)

plt.bar(items,count)
fig=plt.gcf()
fig.set_size_inches(50,50)
plt.xticks(fontsize=4,rotation=90)
plt.show()

import timeit

start = timeit.default_timer()
noOfTransactions=len(data)
allFrequent=[]
frequentItemSet = generateFrequentItemSet(itemOfTrans, noOfTransactions, minSup, data, allFrequent)
associationRules = generateAssociationRule(allFrequent)

AprioriOutput = aprioriOutput(associationRules, data, minConf)


counter = 0
y=1
if len(AprioriOutput) == 0:
    print("There are no association rules for this support and confidence.")
else:
    for i in AprioriOutput:
        if counter == 0:
            print(str(i[0]) + "------>" + str(i[1]), end='  ')
            counter += 1
        else:
            print(i)
            counter=0

stop = timeit.default_timer()

print('Time: ', stop - start)



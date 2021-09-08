import csv
import itertools
import matplotlib.pyplot as plt
def generateFrequentItemSet(CandidateList, noOfTransactions, minimumSupport, dataSet, fatherFrequentArray,itemTID):
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

        generateCandidateSets(dataSet, frequentItemsArray, noOfTransactions, minimumSupport,fatherFrequentArray,itemTID)


def generateCandidateSets(dataSet, frequentItemsArray, noOfTransactions, minimumSupport,fatherFrequentArray,itemTID):
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
    listTID=[]
    tempTID=[]
    for item in arrayAfterCombinations:
        count = 0
        for y in item:
            for z in itemTID:
                if y==z[0][0]:
                    if listTID==[]:
                        listTID=z[2]
                    else:
                        listTID=list(set(listTID).intersection(set(z[2])))
        tempTID.append(listTID)
        temp.append(item)
        temp.append(len(listTID))
        temp.append(listTID)
        candidateSetArray.append(temp)
        temp = []
        listTID=[]


    generateFrequentItemSet(candidateSetArray, noOfTransactions, minimumSupport, dataSet, fatherFrequentArray,itemTID)


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


def Nmaxelements(list, N):
    list1=[]
    for item in list:
        list1.append(item[1])
    # print(list1)
    final_list = []

    for i in range(0, N):
        max1 = 0

        for j in range(len(list1)):
            if list1[j] > max1:
                max1 = list1[j]

        list1.remove(max1)
        final_list.append(max1)

    # print(final_list)
    # for item in list:
    #     for i in final_list:
    #         if i==item[1]:
    #             print(item[0]," ",item[1])
def Nminelements(List, N):
    list1=[]
    for item in List:
        list1.append(item[1])
    final_list = []
    indexList=[]
    index=0
    z=0
    for i in range(0, N):
        min = 10000
        z=0
        for j in range(len(list1)):
            if list1[j] < min:
                min = list1[j]
                index=j
        indexList.append(list1)
        z += 1
        list1.remove(min)
        final_list.append(min)

    # print(final_list)
    final_list=list(dict.fromkeys(final_list))
    for item in List:
        for i in final_list:
            if i==item[1]:
                print(item[0]," ",item[1])


import timeit


start = timeit.default_timer()
i=1
itemOfTrans=[]
minSup=2
minConf=60
frequentItemset=[]
with open('groceries.csv', newline='') as csvfile:
    readFile = csv.reader(csvfile, delimiter=',', quotechar='|')
    data = list(readFile)
    listOfTransactions = {} #dict trans
    for items in data:
        for item in items:

            listOfTransactions.setdefault(item, [])
            listOfTransactions[item].append(i)
        i+=1
    items=[]
    count=[]
    for key, value in listOfTransactions.items():
        temp = [[key], len(value),value]
        itemOfTrans.append(temp)#list transaction with count
        items.append(key)
        count.append(value)

# plt.bar(items,count)
# fig=plt.gcf()
# fig.set_size_inches(50,50)
# plt.xticks(fontsize=4,rotation=90)
# plt.show()
print(itemOfTrans)
Nmaxelements(itemOfTrans,20)
Nminelements(itemOfTrans,20)
noOfTransactions=len(data)
allFrequent=[]
frequentItemSet = generateFrequentItemSet(itemOfTrans, noOfTransactions, minSup, data, allFrequent,itemOfTrans)
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




import matplotlib.pyplot as plt
import csv

# 1

ppvt = []
momage = []

with open('data.csv', 'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        ppvt.append(int(row[1]))
        momage.append(row[3])

momage.sort()

plt.scatter(ppvt, momage, color='b', s=100)
plt.xticks(rotation=25)
plt.xlabel('ppvt')
plt.ylabel('momage')
plt.title('ppvt Report', fontsize=20)

plt.show()

min_epsilon = 0.01
epsilon = 1
epsilon_decay=0.9995
episodes = 30000
counter = 0

while epsilon>=min_epsilon and counter<episodes:
    counter += 1
    epsilon = epsilon*epsilon_decay

print(counter)
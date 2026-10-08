import random

roll = random.choice(['rock', 'paper', 'scissors'])

computer_choice =  'scissors'

user_choice = input("Enter your choice (rock, paper, scissors): ").lower()

if (computer_choice == user_choice):
    print("It's a tie!")
elif (user_choice == "rock" and computer_choice == "scissors") or \
     (user_choice == "paper" and computer_choice == "rock") or \
     (user_choice == "scissors" and computer_choice == "paper"):
    print("You win!")
else:
    print("Computer wins!")

# acronyms = []

# for i in range(0, 7, 1):
#     acronym = input("Enter an acronym: ")
#     acronyms.append(acronym)

# print("The acronyms you entered are:")
# for acronym in acronyms:
#     print(acronym)

import requests

response = requests.get('https://api.open-notify.org/astros.json')
json = response.json()
print("Number of people in space:", json['number'])   
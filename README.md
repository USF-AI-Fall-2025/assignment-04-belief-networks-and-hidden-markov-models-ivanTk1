# Belief-Networks-Hidden-Markov-Models
Fall 2025 CS 362/562

Question 1: Input: i eat a hamburger daly → Output: e eat d hamburger daly
In thi example I was changed to an e and a was "corrected" to d. This happened becayse single letter wordsa are especially hard to correct. the start/end transitions and smoothed emission probabilities can outweigh the (tiny) evidence to not change it. Basicly the model is overfitting the letter tranition frequency. 

Question 2: 
nevade → nevede (intended: nevada)
equire → iquere (intended: acquire)
The problem here is that it doesnt know how to fix mistakes when one letter is replaced with a different in the same spot. the model never learned them because it only compared letters that line up. So it guesses the closest thing it knows based on letter patterns it has seen before, which often gives the wrong word.

Question 3: teo → ten
This one worked becasue it fit a common pattern that the model had already seen many times. It was able to recongnize that teo was close to ten and ten made more sense based on its training data.

Questionm 4: If the training data came from real typos that people actually made online, the model would learn patterns, like when people accidentally hit nearby keys or forget double letters. This would obviously make it much more accurate. if the data was random it would learn random mistakes that people noramly dont make making it worse at correcting real data. 

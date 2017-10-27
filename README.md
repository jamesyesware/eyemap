# eyemap
### Machine Learning Powered Auto-Email Detection

___

Machine learning is cool and hip and popular, so why *wouldn't* we do something with it? Our algorithm works to classify an email
as being either automatically generated (an out-of-office reply, for example) or genuine, which is to say written by a person. The only
data point we use is the subject of the email.

You can train our model and test it against some seperate emails to see what kind of results are possible. In our first tests, we were routinely
seeing 99.99% accuracy on arbitrary test input. Wow! What a time to be alive.

```python
from classify import Classifier

classifier = Classifier()
classifier.start_training(50000) # no. of epochs

test_subjects = [<list of real subjects that you did not train on!>]

classifier.practice_run(test_subjects)
```

That's the simplest way to see the model work. We will be updating the API to increase the usability/testability.

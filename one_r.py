import pandas as pd
from sklearn.model_selection import train_test_split


class ConfusionMatrix:

    def __init__(self, actual, predicted):
        self.actual = actual
        self.predicted = predicted
        self.labels = sorted(set(actual + predicted))
        self.matrix = self.build_matrix()
    
    def build_matrix(self):
        matrix = {label: {label: 0 for label in self.labels} for label in self.labels}

        for act, pred in zip(self.actual, self.predicted):
            matrix[act][pred] += 1
        
        return matrix
    def print_matrix(self):
        
        
        print("CONFUSION MATRIX:")
        print(" " * 15 + "Predicted as")
        print(" " * 12 + " ".join(f"{label:>5}" for label in self.labels))
        print(" " * 10 + "-" * (6 * len(self.labels)))

        # Printing rows with actual labels
        for label in self.labels:
            row = " ".join(f"{self.matrix[label][l]:>5}" for l in self.labels)
            print(f"Actual {label:>4} | {row}")


class OneR:
    def __init__(self):
        self.attribute_rules = {}
        self.best_rules = {}
    
    def classify(self, instance):
        for best_attribute, best_rule in self.best_rules.items():
            attribute_val = instance[best_attribute]
            for attribute_value, classification in best_rule.rules.items():
                if attribute_val == attribute_value:
                    return classification.classification
    
    def show_best_rules(self):
        for attribute, attribute_rules in self.best_rules.items():
            for attribute_value, classification in attribute_rules.rules.items():
                print(f'RULE: if {attribute} = {attribute_value}, then classify as {classification.classification}. Accuracy: {classification.correct}/{classification.total}')
    
    def __str__(self):
        return f'{self.attribute_rules}'

class AttributeRules:
    def __init__(self):
        self.rules = {}

    def __str__(self):
        return f'{self.rules}'

class Classification:
    def __init__(self, attribute_value, classification, correct, total):
        self.correct = correct
        self.total = total
        self.attribute_value = attribute_value
        self.classification = classification

    def __str__(self):
        return f'{self.attribute_value} as {self.classification} {self.correct}/{self.total}'

CLASS_NAME = 'Outcome'
TEST_INSTANCES = None

def get_best_rules(one_r):
    best_rules = {}
    best_accuracy = 0
    for attribute, attribute_rules in one_r.attribute_rules.items():
        accuracy_numerator = 0
        accuracy_denominator = 0
        for attribute_value, classification in attribute_rules.rules.items():
            accuracy_numerator += classification.correct
            accuracy_denominator += classification.total
        accuracy = accuracy_numerator / accuracy_denominator
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_rules = {attribute: attribute_rules}
    return best_rules


def build_rules(data_location, one_r):
    df = pd.read_csv(data_location)
    

    df['Glucose'] = pd.cut(df['Glucose'], bins=4, labels=False)
    df['BloodPressure'] = pd.cut(df['BloodPressure'], bins=10, labels=False)
    df['SkinThickness'] = pd.cut(df['SkinThickness'], bins=5, labels=False)
    df['BMI'] = pd.cut(df['BMI'], bins=5, labels=False)
    df['Age'] = pd.cut(df['Age'], bins=5, labels=False)
    df['Insulin'] = pd.cut(df['Insulin'], bins=5, labels=False)
    df['DiabetesPedigreeFunction'] = pd.cut(df['DiabetesPedigreeFunction'], bins=5, labels=False)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=1)

    global TEST_INSTANCES
    #make TEST_INSTANCES a list of all instances in df
    TEST_INSTANCES = test_df.to_dict(orient='records')

    attributes = [col for col in train_df.columns if col != CLASS_NAME]

    for attribute in attributes:
        attribute_rules = AttributeRules()
        attribute_values = train_df[attribute].dropna().unique()

        for attribute_value in attribute_values:
            class_counts = (
                train_df[train_df[attribute] == attribute_value][CLASS_NAME]
                .value_counts()
                .to_dict()
            )
            most_frequent_class = (
                train_df[train_df[attribute] == attribute_value][CLASS_NAME]
                .mode()
                .iloc[0]
            )

            total_count = train_df[train_df[attribute] == attribute_value].shape[0]
            most_frequent_class_count = class_counts[most_frequent_class]
            classification = Classification(attribute_value, most_frequent_class, most_frequent_class_count, total_count)
            attribute_rules.rules[attribute_value] = classification

        one_r.attribute_rules[attribute] = attribute_rules
    
    one_r.best_rules = get_best_rules(one_r)

def classify_instances(one_r, instances):
    correct = 0
    total = 0
    actuals = []
    predicted = []
    for instance in instances:
        classification = one_r.classify(instance)
        if classification == instance[CLASS_NAME]:
            correct += 1
        total += 1
        actuals.append(instance[CLASS_NAME])
        predicted.append(classification)
    confusion_matrix = ConfusionMatrix(actuals, predicted)
    confusion_matrix.print_matrix()
    print()
    print(f"Total Test Instances: {total}")
    print(f"Correctly Classified: {correct} ({correct / total * 100}%)")
    
    




def main():
    one_r = OneR()

    data_location = '../data/diabetes.csv'

    build_rules(data_location, one_r)

    classify_instances(one_r, TEST_INSTANCES)

    one_r.show_best_rules()
   

if __name__ == "__main__":
    main()

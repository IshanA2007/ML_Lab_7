import naive_bayes as nb
import one_r as oner
import imported_naive_bayes as inb
import imported_one_r as ioner
from prettytable import PrettyTable

def print_probability_table(probability_tables):
    for class_label, class_dict in probability_tables.items():
        print(f"\nClass: {class_label}")
        
        for attribute, probabilities in class_dict.classDictionary.items():
            table = PrettyTable()
            table.field_names = ["Value", "Probability"]
            
            for value, prob in probabilities.attributeDictionary.items():
                table.add_row([value, f"{prob:.4f}"])
            
            print(f"  Attribute: {attribute}")
            print(table)

def run_nb(data_location):
    
    print("-------------------------")
    print("RUNNING NAIVE BAYES CLASSIFIER")
    nb.build_data(data_location)
    print("-------------------------")
    print()

    print("-------------------------")
    print("PROBABILITY TABLES CREATED")
    print("-------------------------")
    print()
    print_probability_table(nb.P_TABLE.classTables)
    print()
    print("-------------------------")
    print("CLASSIFYING INSTANCES")
    print("-------------------------")
    print()

    nb.classify_instances(nb.TEST_INSTANCES)

    print()
    print("-------------------------")
    print("END OF NAIVE BAYES CLASSIFIER")
    print("-------------------------")

def run_inb(data_location):
    print("-------------------------")
    print("RUNNING IMPORTED NAIVE BAYES CLASSIFIER")
    print("-------------------------")
    print()

    inb.classify_instances(data_location)

    print()
    print("-------------------------")
    print("END OF IMPORTED NAIVE BAYES CLASSIFIER")
    print("-------------------------")

def run_oner(data_location):
    print("-------------------------")
    print("RUNNING ONE-R CLASSIFIER")
    print("-------------------------")
    print()

    one_r = oner.OneR()
    oner.build_rules(data_location, one_r)
    one_r.show_best_rules()

    print()
    print("-------------------------")
    print("CLASSIFYING INSTANCES")
    print("-------------------------")
    print()

    oner.classify_instances(one_r, oner.TEST_INSTANCES)

    print()
    print("-------------------------")
    print("END OF ONE-R CLASSIFIER")
    print("-------------------------")

def run_ioner(data_location):
    print("-------------------------")
    print("RUNNING IMPORTED ONE-R CLASSIFIER")
    print("-------------------------")
    print()

    ioner.classify_instances(data_location)

    print()
    print("-------------------------")
    print("END OF IMPORTED ONE-R CLASSIFIER")
    print("-------------------------")

def main():
    data_location = "../data/diabetes.csv"
    run_nb(data_location)
    run_inb(data_location)
    run_oner(data_location)
    run_ioner(data_location)


if __name__ == "__main__":
    main()
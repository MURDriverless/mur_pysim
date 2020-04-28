import sys
from workspaces.alex import main as alex
from workspaces.demi import main as demi
from workspaces.dennis import main as dennis
from workspaces.joseph import main as joseph
from play_as_human import play_as_human


def sanitise_name(name):
    # Remove "--ws=" from the argument
    name = name.replace("--ws=", "")
    # Set everything as lowercase
    name = name.lower()
    return name


def print_help():
    print("\nUsage: python main.py 'modifier' \n")
    print("Modifiers:")
    print("-h, --help : Prints this help screen")
    print("--ws='workspace_name' : Runs main.py found in the specified workspace")
    print("               names -> alex, demi, dennis, joseph")
    print("--human : Play using your arrow keys")
    print("        : Up for accelerate, down for brakes, and left and right for steering \n")


def main(argv):
    # If the length of CLI command is not 2, that means it's a wrong usage. Print help instead
    if len(argv) is not 2:
        print_help()
        return

    modifier = argv[1].lower()

    if "--human" in modifier:
        return play_as_human()

    elif "-h" in modifier or "--help" in modifier:
        return print_help()

    elif "--ws=" in modifier:
        # Perform operations to get the name of the workspace
        name = sanitise_name(argv[1])

        if name == "alex":
            return alex.main()

        elif name == "demi":
            return demi.main()

        elif name == "dennis":
            return dennis.main()

        elif name == "joseph":
            return joseph.main()

        else:
            print(f"Workspace named {name} cannot be found\n")

    else:
        return print_help()


if __name__ == "__main__":
    main(sys.argv)

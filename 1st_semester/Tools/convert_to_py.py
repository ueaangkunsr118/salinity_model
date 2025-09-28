import subprocess
import os

FILENAME = "wetergate3.ipynb"


# Convert the notebook to a Python script
def convert(filename):
    # Remove old Python script
    if os.path.exists(filename.replace(".ipynb", ".py")):
        os.remove(filename.replace(".ipynb", ".py"))

    # Remove old converted Python script
    if os.path.exists("generated_" + filename.replace(".ipynb", ".py")):
        os.remove("generated_" + filename.replace(".ipynb", ".py"))

    # Convert the notebook to a Python script
    subprocess.run(["jupyter", "nbconvert", "--to", "script", filename])

    # Read the converted script
    with open(filename.replace(".ipynb", ".py"), "r") as f:
        lines = f.readlines()
        f.close()

    os.remove(filename.replace(".ipynb", ".py"))

    # Remove lines containing 'get_ipython()'
    lines = [line for line in lines if "get_ipython()" not in line and "#" not in line]
    lines = [line for line in lines if line.replace(" ", "").replace("\n", "") != ""]

    # Separate import statements and other lines
    imports = [
        line for line in lines if line.startswith("import") or line.startswith("from")
    ]
    others = [
        line
        for line in lines
        if not line.startswith("import") and not line.startswith("from")
    ]

    # Write the modified script
    with open("generated_" + filename.replace(".ipynb", ".py"), "w") as f:
        f.write("".join(imports))
        f.write("\ndef main():\n")
        f.write(
            "".join("    " + line if line != "\n" else line for line in others)
        )  # Indent non-empty lines
        f.write('\nif __name__ == "__main__":\n')
        f.write("    main()\n")
        f.close()


if __name__ == "__main__":
    convert(FILENAME)

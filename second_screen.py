from tkinter import *
from tkinter import messagebox
import third_screen
import json

prcs = None
buttons = []
valid = [True]
newwin = None
data_lst = {}


def sec_main():
    root = Tk()
    root.title('CPU Scheduling')

# x = (root.winfo_screenwidth() - root.winfo_reqwidth()) / 2
# y = (root.winfo_screenheight() - root.winfo_reqheight()) / 2
# root.geometry("+%d+%d" % (x, y))
    root.geometry('330x130+550+200')
    root.resizable(0, 0)
    frame = Frame(root)
    # f=Frame(root, height=50, bd=8, relief="raise")
    frame.grid(row=0, column=0)

    def submit():
        f = open('fl.txt')
        txt = next(f)
        txt = int(txt.strip('\n'))
        f.close()
        dct = {}
        for i in range(0, len(buttons), 3):
            try:
                dct['id'] = buttons[i][0].get()
                if txt == 4:
                    dct['Period'] = int(buttons[i+1][0].get())
                    dct['WCET'] = int(buttons[i+2][0].get())
                if txt==2:
                    dct['dt'] = int(buttons[i+2][0].get())
            except:
                valid[0] = False
            else:
                valid[0] = True
            data_lst[buttons[i][0].get()] = dct
            dct = {}
        if not valid[0]:
            messagebox.showinfo("Input Error", "Please enter all the values")
            reset()
        else:
            data_f = open('data.json', 'w')
            data_f.write(json.dumps(data_lst))
            data_f.close()
            root.destroy()
            third_screen.main()

            return

    def reset():
        for i in range(len(buttons)):
            buttons[i][0].delete(0, END)

    def table_win():
        f = open('fl.txt')
        txt = next(f)
        txt = int(txt.strip('\n'))
        f.close()
        if txt == 6:
            f = open('fl.txt', 'a')
            f.write('\n'+prcs1.get()+'\n')
        global newwin
        processes = int(prcs.get())
        newwin = Toplevel(root)
        b = Label(newwin, text="Process")
        b.grid(row=0, column=0)
        print(txt)
        if txt == 1:
            b = Label(newwin, text="DeadLine")
            b.grid(row=0, column=1)
            column = 3
        else:
            if txt !=4:
                b = Label(newwin, text="Arrival Time")
                b.grid(row=0, column=1)
                b = Label(newwin, text="DeadLine")
                b.grid(row=0, column=1)
        column = 3
        if txt == 4:
            b = Label(newwin, text="Period")
            b.grid(row=0, column=1)
        b = Label(newwin, text="worst-case computation Time")
        b.grid(row=0, column=2)
        for i in range(1, processes+1):  # Rows
            for j in range(column):
                tmp = [Entry(newwin, text=""), i, j]
                buttons.append(tmp)
        for i in buttons:
            i[0].grid(row=i[1], column=i[2])

        sub = Button(newwin, text="Submit", command=submit).grid(
            row=processes+2, column=1)
        Reset = Button(newwin, text="Reset", command=reset).grid(
            row=processes+3, column=1)

    Label(frame, text="Enter Processes").grid(row=0, column=1)
    f = open('fl.txt')
    txt = next(f)
    txt = int(txt.strip('\n'))
    f.close()
    prcs1 = None
    if txt == 6:
        Label(frame, text="Enter Time Quantum").grid(row=1, column=1)
        prcs1 = Entry(frame, text="")
        prcs1.grid(row=1, column=2)
    prcs = Entry(frame, text="")
    prcs.grid(row=0, column=2)

    Label(frame, text="").grid(row=2, column=1)
    Label(frame, text="").grid(row=3, column=1)
    # Label(frame, text="").grid(row=4, column=1)
    generate = Button(frame, text="Generate", padx=6, pady=6, fg="black",
                      font=('arial', 12, 'bold'), width=16, height=1,
                      command=table_win).grid(row=5, column=1)
    exit_button = Button(frame, text="Exit", padx=6, pady=6, fg="black", font=(
        'arial', 12, 'bold'), width=16, height=1, command=root.destroy).grid(row=5, column=2)

    root.mainloop()


if __name__ == "__main__":
    print("Run cpu_sheduler.py")
    sec_main()

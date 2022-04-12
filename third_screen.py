from tkinter import *
import connector
import pandas as pd
import matplotlib.pyplot as plt

def gchart(data):
    data = pd.DataFrame(data)
    plt.barh(y=data["id"], left=data["st"], width=data["ct"])
    plt.xlabel("time")
    plt.ylabel("process")
    plt.title("grant chart")
    plt.show()
    


def main():
    root = Tk()
    root.title("CPU Scheduling")
    root.geometry('500x250+550+200')
    root.resizable(0, 0)
    Tops = Frame(root, height=50, bd=8, relief="flat")
    Tops.grid(row=5)
    Label(Tops, font=('arial', 30, 'bold'),
          text="  Results  ", bd=5).grid(row=0, column=0)

    f1 = Frame(root, height=10, width=10, bd=4, relief="flat")
    f1.grid(row=6)
    av, fesible = connector.main()
    av_fes = " Feasible = {}".format(fesible)
    Label(f1, font=('arial', 15, 'bold'),
          text=av_fes, bd=5).grid(row=0, column=0)

    e = Entry(root, fg='blue',
                               font=('Arial',16,'bold'))
    
    e.grid(row=7,columnspan=2, padx=10, pady=(10,4), sticky="nsew")
    e.insert(END, "process")
    for i in range(len(av)):
        e.grid(row=7,columnspan=2, padx=10, pady=(10,4), sticky="nsew")
        e.insert(END, av[i]["id"])
    
    e = Entry(root, fg='red',
                               font=('Arial',16,'bold'))
    e.grid(row=8,columnspan=2, padx=10, pady=(10,4), sticky="nsew")
    e.insert(END,"lateness")
    for i in range(len(av)):
        e.grid(row=8,columnspan=2, padx=10, pady=(10,4), sticky="nsew")
        e.insert(END, av[i]["latness"])

    f2 = Frame(root, height=10, width=100, bd=4, relief="flat")
    f2.grid(row=55)
   
    btnchart = Button(f2, text="Gantt Chart", padx=6, pady=6, bd=2, fg="black",
                      font=('arial', 12, 'bold'), width=14, height=1,
                      command=lambda: gchart(av)).grid(row=0, column=0)

    root.mainloop()


if __name__ == "__main__":
    print("Run cpu_sheduler.py")
    main()

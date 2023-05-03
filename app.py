import inspect
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfile
from filterable_interface import FilterableInterface
from data_filterer import DataFilterer
from tkinter import messagebox


class CustomText(tk.Text):
    def __init__(self, *args, placeholder="", hard_placeholder="", **kwargs):
        """A text widget that report on internal widget commands"""
        tk.Text.__init__(self, *args, **kwargs)

        # create a proxy for the underlying widget
        self._orig = self._w + "_orig"
        self.tk.call("rename", self._w, self._orig)
        self.tk.createcommand(self._w, self._proxy)

        self.insert("1.0", hard_placeholder)

        if placeholder:
            self.insert("1.0", placeholder)
            self.bind("<FocusIn>", self.foc_in)

    def foc_in(self, *args):
        self.delete('0.0', 'end')

    def _proxy(self, command, *args):
        cmd = (self._orig, command) + args
        result = self.tk.call(cmd)

        if command in ("insert", "delete", "replace"):
            self.event_generate("<<TextModified>>")

        return result


class app:
    def __init__(self):
        self.chosen_class_name: str = None
        self.filterable_instance: FilterableInterface = None
        self.filterer_instance: DataFilterer = None
        self.semantic_percent: float = 0.0
        self.outlier_percent: float = 0.0
        self.filter_downscale_dim: int = 100
        self.chosen_layer: str = None
        self.dataset_items = None
        self.dataset_batches = None
        self.feature_map_shape = (0,)
        self.output_path: str = "out.txt"


        if 0:  # todo remove this
            import Resnet32
            self.filterable_instance = Resnet32.Resnet32()
            self.chosen_class_name = "Resnet32"
            self.chosen_layer = "module.avgpool"
            self.filterer_instance = DataFilterer(model=self.filterable_instance, layer=self.chosen_layer)
            self.dataset_items, self.dataset_batches = 50000, 391

        self.master = tk.Tk()
        self.master.geometry("600x600")
        self.main_screen()

    def get_frame(self):
        for i in self.master.winfo_children():
            i.destroy()
        frame = tk.Frame(self.master)
        frame.pack(fill="both", expand=True)
        return frame

    def main_screen(self):
        frame = self.get_frame()

        self.add_model_info(frame, start_row=0)

        if self.filterable_instance is None:
            return

        self.add_layer_info(frame, start_row=1)

        if self.chosen_layer is None:
            return

        # Dataset info
        self.add_dataset_info(frame, start_row=2)

        # Filtering config
        self.add_filtering_config(frame, start_row=6)

        # Filtering button
        self.add_filtering_button(frame, start_row=10)

    def add_model_info(self, frame, start_row):
        info_text1 = ttk.Label(frame, text=f"Imported model: {self.chosen_class_name}")
        import_model_btn = ttk.Button(frame, text="Choose", command=self.import_model)

        info_text1.grid(row=start_row, column=0, rowspan=1, columnspan=2, sticky="W")
        import_model_btn.grid(row=start_row, column=2, rowspan=1, columnspan=1)

    def add_layer_info(self, frame, start_row):
        info_text2 = ttk.Label(frame, text=f"Layer to extract features: {self.chosen_layer}")
        choose_layer_btn = ttk.Button(frame, text="Choose", command=self.choose_layer)

        info_text2.grid(row=start_row, column=0, rowspan=1, columnspan=2, sticky="W")
        choose_layer_btn.grid(row=start_row, column=2, rowspan=1, columnspan=1)

    def add_dataset_info(self, frame, start_row):

        def set_dataset_info():
            self.dataset_items, self.dataset_batches = self.filterer_instance.get_dataset_info()
            self.main_screen()

        info_text3 = ttk.Label(frame, text=f"Filterable dataset overview")

        info_text4 = ttk.Label(frame, text=f"Items")
        info_text5 = ttk.Label(frame, text=f"Batches")
        info_text6 = ttk.Label(frame, text=f"Feature map shape")

        info_text7 = ttk.Label(frame, text=f"{self.dataset_items} ")
        info_text8 = ttk.Label(frame, text=f"{self.dataset_batches}")
        info_text9 = ttk.Label(frame, text=f"{self.filterer_instance.get_feature_map_shape()}")

        frame.rowconfigure(start_row, minsize=20)
        info_text3.grid(row=start_row + 1, column=0, rowspan=1, columnspan=3)

        if None in (self.dataset_items, self.dataset_batches):
            calc_dataset_btn = ttk.Button(frame, text="Get dataset overview", command=set_dataset_info)
            calc_dataset_btn.grid(row=start_row + 2, column=0, rowspan=1, columnspan=3)
            return

        info_text4.grid(row=start_row + 2, column=0, rowspan=1, columnspan=1)
        info_text5.grid(row=start_row + 2, column=1, rowspan=1, columnspan=1)
        info_text6.grid(row=start_row + 2, column=2, rowspan=1, columnspan=1)

        info_text7.grid(row=start_row + 3, column=0, rowspan=1, columnspan=1)
        info_text8.grid(row=start_row + 3, column=1, rowspan=1, columnspan=1)
        info_text9.grid(row=start_row + 3, column=2, rowspan=1, columnspan=1)

    def add_filtering_config(self, frame, start_row):

        info_text10 = ttk.Label(frame, text=f"Filtering configuration")
        info_text11 = ttk.Label(frame, text=f"Semantic similarities:")
        info_text12 = ttk.Label(frame, text=f"Outliers:")

        info_text13 = ttk.Label(frame, text=f"")
        info_text14 = ttk.Label(frame, text=f"")

        info_text15 = ttk.Label(frame, text=f"")

        semantic_percent_tbox = CustomText(frame, height=1, width=10,
                                           placeholder=str(self.semantic_percent * 100) + "%")
        outlier_percent_tbox = CustomText(frame, height=1, width=10, placeholder=str(self.outlier_percent * 100) + "%")

        def settext():
            i1 = int(self.semantic_percent * self.dataset_items)
            i2 = int(self.outlier_percent * self.dataset_items)
            info_text13.config(text=f"{i1} removed")
            info_text14.config(text=f"{i2} removed")
            info_text15.config(text=f"Resulting dataset size: {self.dataset_items - i1 - i2}")

        def onModification_sem(event):
            percentage = event.widget.get("1.0", "end-1c")
            try:
                percentage = float(percentage.strip())
            except:
                percentage = 0.0
            percentage = min(100.0, percentage)
            percentage = max(0.0, percentage) / 100
            self.semantic_percent = percentage
            if None not in (self.dataset_items, self.dataset_batches):
                settext()

        def onModification_out(event):
            percentage = event.widget.get("1.0", "end-1c")
            try:
                percentage = float(percentage.strip())
            except:
                percentage = 0.0
            percentage = min(100.0, percentage)
            percentage = max(0.0, percentage) / 100
            self.outlier_percent = percentage
            if None not in (self.dataset_items, self.dataset_batches):
                settext()

        semantic_percent_tbox.bind("<<TextModified>>", onModification_sem)
        outlier_percent_tbox.bind("<<TextModified>>", onModification_out)
        if None not in (self.dataset_items, self.dataset_batches):
            settext()

        frame.rowconfigure(start_row, minsize=20)
        info_text10.grid(row=start_row + 1, column=0, rowspan=1, columnspan=4)

        info_text11.grid(row=start_row + 2, column=0, rowspan=1, columnspan=1, sticky="E")
        semantic_percent_tbox.grid(row=start_row + 2, column=1, rowspan=1, columnspan=1, sticky="W")
        info_text13.grid(row=start_row + 2, column=2, rowspan=1, columnspan=2, sticky="W")

        info_text12.grid(row=start_row + 3, column=0, rowspan=1, columnspan=1, sticky="E")
        outlier_percent_tbox.grid(row=start_row + 3, column=1, rowspan=1, columnspan=3, sticky="W")
        info_text14.grid(row=start_row + 3, column=2, rowspan=1, columnspan=2, sticky="W")

        info_text15.grid(row=start_row + 4, column=0, rowspan=1, columnspan=3)

    def add_filtering_button(self, frame, start_row):

        def filter():
            idxs = self.filterer_instance.get_idxs(outliar_percentage=self.outlier_percent,
                                                   semantic_percentage=self.semantic_percent)
            with open(self.output_path, "w") as f:
                for arg in idxs:
                    f.write(str(arg) + "\n")

        def change_path(event):
            self.output_path = event.widget.get("1.0", "end-1c")

        info_text1 = ttk.Label(frame, text=f"Output file path:")
        output_folder_tbox = CustomText(frame, height=1, width=20, placeholder=self.output_path)
        filter_btn = ttk.Button(frame, text="Filter", command=filter)

        output_folder_tbox.bind("<<TextModified>>", change_path)

        frame.rowconfigure(start_row, minsize=20)
        info_text1.grid(row=start_row + 1, column=0, rowspan=1, columnspan=1)
        output_folder_tbox.grid(row=start_row + 2, column=0, rowspan=1, columnspan=2)
        filter_btn.grid(row=start_row + 2, column=3, rowspan=1, columnspan=1)

    def choose_layer(self):
        frame = self.get_frame()
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(2, weight=1)
        instruction_label = ttk.Label(frame, text="Choose the layer which is used to extract feature maps from")
        tree = ttk.Treeview(frame)

        def get_model_layers(model, location, prefix=""):
            names = [a for a in model.named_children()]
            for i, child in enumerate(model.children()):
                layer_name = names[i][0]

                name = type(child).__name__
                has_children = len([a for a in child.children()]) > 0
                if not has_children:
                    name = child.__str__()
                treenode = tree.insert(parent=location, index="end", tags=prefix + "." + layer_name, text=name)
                if has_children:
                    get_model_layers(child, treenode, prefix=prefix + "." + layer_name)

        get_model_layers(self.filterable_instance.get_model(), "")

        chosen_label = ttk.Label(frame, text="chosen layer: None")

        def chooseBtn():
            self.chosen_layer = tree.item(tree.focus())["tags"][0][1:]
            self.filterer_instance = DataFilterer(model=self.filterable_instance, layer=self.chosen_layer, device="cuda")
            self.feature_map_shape = self.filterer_instance.get_feature_map_shape()
            self.main_screen()

        choose_layer_btn = ttk.Button(frame, text="Choose", command=chooseBtn)
        choose_layer_btn["state"] = "disabled"

        def selectItem(a):
            layer_name = tree.item(tree.focus())["tags"][0][1:]
            df = DataFilterer(model=self.filterable_instance, layer=layer_name)
            output_shape = df.get_feature_map_shape()
            df.__del__()
            chosen_label.config(text=f"""chosen layer: {layer_name}\nLayer output shape: {output_shape}""")
            choose_layer_btn["state"] = "disabled" if not output_shape else "normal"

        tree.bind('<ButtonRelease-1>', selectItem)

        # put items as a grid
        instruction_label.grid(row=0, column=0, sticky="NESW", columnspan=2)
        chosen_label.grid(row=1, column=0, sticky="E")
        choose_layer_btn.grid(row=1, column=1, sticky="E")
        tree.grid(row=2, column=0, sticky="NESW", rowspan=5, columnspan=2)

    def import_model(self):
        file = askopenfile(mode='r', filetypes=[('Python Files', '*.py')])

        if file is None:
            return

        import importlib.util
        spec = importlib.util.spec_from_file_location("database", file.name)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        found_classes = []
        for a, c in inspect.getmembers(foo, inspect.isclass):
            upper_classes = [s.__name__ for s in c.__bases__]
            if "FilterableInterface" in upper_classes:
                found_classes.append((a, c))

        def choose(a):
            try:
                self.filterable_instance = a[1]()
                self.chosen_class_name = a[0]
                self.chosen_layer = None
            except Exception as err:
                messagebox.showerror("Error", f"Error initializing found class:\n{err}")
            self.main_screen()

        if len(found_classes) == 0:
            messagebox.showerror("Error", "A class, that extends the FilterableInterface not found")
            self.main_screen()
            return

        if len(found_classes) == 1:
            choose(found_classes[0])
            return

        # multiple classes with FilterableInterface found in the file
        frame = self.get_frame()

        label = ttk.Label(frame, text="Multiple classes with FilterableInterface found. Choose the class which is used")
        label.pack()

        for a in found_classes:
            btn = ttk.Button(frame, text=a[0], command=lambda: choose(a))
            btn.pack()


if __name__ == "__main__":
    app = app()
    app.master.attributes('-type', 'dialog')
    app.master.mainloop()

    # root = tk.Tk()
    # root.title('Dataset filterer application')
    #
    # turn_on = tk.Button(root, text="Import model", command=import_model)
    # turn_on.pack()
    #
    # turn_off = tk.Button(root, text="OFF", command=root.quit)
    # turn_off.pack()
    #
    # root.mainloop()

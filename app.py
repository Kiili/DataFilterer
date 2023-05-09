import inspect
import math
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.filedialog import askopenfile
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import PIL
from PIL import ImageTk, Image

from filterable_interface import FilterableInterface
from data_filterer import DataFilterer


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
        self.filter_downscale_dim: int | None = None  # todo optionally set this by the user
        self.chosen_layer: str = None
        self.dataset_items = None
        self.dataset_batches = None
        self.feature_map_shape = (0,)
        self.output_path: str = "out.txt"
        self.downscale_method = "PCA"

        self.master = tk.Tk()
        self.master.minsize(400, 400)

        # todo remove these
        if 0:        # quick setup for testing
            import examples.resnet32
            self.filterable_instance = examples.resnet32.Resnet32Example()
            self.chosen_class_name = "Resnet32Example"
            self.chosen_layer = "avgpool"
            self.semantic_percent = 0.05
            self.outlier_percent = 0.05
            self.filterer_instance = DataFilterer(self.filterable_instance, layer=self.chosen_layer)
            self.dataset_items, self.dataset_batches = self.filterer_instance.get_dataset_info()
            self.filterer_instance.get_idxs(semantic_percentage=self.semantic_percent, outlier_percentage=self.outlier_percent, downscale_dim=None, downscale_method=self.downscale_method)
        if 0:        # quick setup for testing
            import examples.yolov5
            self.filterable_instance = examples.yolov5.Yolov5Example()
            self.chosen_class_name = "Yolov5Example"
            self.chosen_layer = "model.model.model.10"
            self.semantic_percent = 0.1
            self.outlier_percent = 0.1
            self.filterer_instance = DataFilterer(self.filterable_instance, layer=self.chosen_layer)
            self.dataset_items, self.dataset_batches = self.filterer_instance.get_dataset_info()
            self.filterer_instance.get_idxs(semantic_percentage=self.semantic_percent, outlier_percentage=self.outlier_percent, downscale_dim=None, downscale_method=self.downscale_method)

        self.main_screen()

    def get_frame(self):
        for i in self.master.winfo_children():
            i.destroy()
        frame = tk.Frame(self.master)
        frame.pack(fill="both", expand=True, padx=(10, 10), pady=(10, 10))
        return frame

    def main_screen(self):
        frame = self.get_frame()

        exit_btn = ttk.Button(frame, text="X", width=3, command=lambda: exit())
        exit_btn.grid(row=0, column=99, sticky="E")

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

        if self.filterer_instance.plot_points is None:
            return

        # View images and plot points buttons
        self.add_visualization_buttons(frame, start_row=13)

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

        info_text0 = ttk.Label(frame, text=f"Filtering configuration")
        info_text1 = ttk.Label(frame, text=f"Semantic similarities:")
        info_text2 = ttk.Label(frame, text=f"Outliers:")

        info_text3 = ttk.Label(frame, text=f"")
        info_text4 = ttk.Label(frame, text=f"")

        info_text5 = ttk.Label(frame, text=f"")

        method = tk.StringVar()
        def choose_method():
            self.downscale_method = method.get()

        choose_downscale_method_menu = ttk.OptionMenu(frame, method, self.downscale_method, "PCA", "UMAP", "T-SVD", "SRP", "GRP", command=lambda a: choose_method())

        semantic_percent_tbox = CustomText(frame, height=1, width=10,
                                           placeholder=str(self.semantic_percent * 100))
        outlier_percent_tbox = CustomText(frame, height=1, width=10,
                                          placeholder=str(self.outlier_percent * 100))

        def settext():
            i1 = int(self.semantic_percent * self.dataset_items)
            i2 = int(self.outlier_percent * self.dataset_items)
            info_text3.config(text=f"{i1} removed")
            info_text4.config(text=f"{i2} removed")
            info_text5.config(text=f"Resulting dataset size: {max(0, self.dataset_items - i1 - i2)}")

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
        info_text0.grid(row=start_row + 1, column=0, rowspan=1, columnspan=4)

        info_text1.grid(row=start_row + 2, column=0, rowspan=1, columnspan=1, sticky="E")
        semantic_percent_tbox.grid(row=start_row + 2, column=1, rowspan=1, columnspan=1, sticky="W")
        info_text3.grid(row=start_row + 2, column=2, rowspan=1, columnspan=2, sticky="W")

        choose_downscale_method_menu.grid(row=start_row + 2, column=4, rowspan=1, columnspan=1)

        info_text2.grid(row=start_row + 3, column=0, rowspan=1, columnspan=1, sticky="E")
        outlier_percent_tbox.grid(row=start_row + 3, column=1, rowspan=1, columnspan=3, sticky="W")
        info_text4.grid(row=start_row + 3, column=2, rowspan=1, columnspan=2, sticky="W")

        info_text5.grid(row=start_row + 4, column=0, rowspan=1, columnspan=3)

    def add_filtering_button(self, frame, start_row):

        def filter():
            idxs = self.filterer_instance.get_idxs(outlier_percentage=self.outlier_percent,
                                                   semantic_percentage=self.semantic_percent,
                                                   downscale_dim=self.filter_downscale_dim,
                                                   downscale_method=self.downscale_method)
            with open(self.output_path, "w") as f:
                for arg in idxs:
                    f.write(str(arg) + "\n")
            self.main_screen()

        def change_path(event):
            self.output_path = event.widget.get("1.0", "end-1c")

        info_text1 = ttk.Label(frame, text=f"Output file path:")
        output_folder_tbox = CustomText(frame, height=1, width=20, placeholder=self.output_path)
        filter_btn = ttk.Button(frame, text="Filter", command=filter)

        output_folder_tbox.bind("<<TextModified>>", change_path)

        frame.rowconfigure(start_row, minsize=20)
        info_text1.grid(row=start_row + 1, column=0, rowspan=1, columnspan=1)
        output_folder_tbox.grid(row=start_row + 2, column=0, rowspan=1, columnspan=2)
        filter_btn.grid(row=start_row + 2, column=2, rowspan=1, columnspan=1)

    def add_visualization_buttons(self, frame, start_row):
        view_imgs_button = ttk.Button(frame, text="View images", command=self.view_images)
        plot_dataset_button = ttk.Button(frame, text="Plot dataset", command=self.visualize_dataset)

        frame.rowconfigure(start_row, minsize=20)
        view_imgs_button.grid(row=start_row + 1, column=0, rowspan=1, columnspan=1)
        plot_dataset_button.grid(row=start_row + 1, column=1, rowspan=1, columnspan=1)

    def visualize_dataset(self):
        from matplotlib import pyplot as plt
        frame = self.get_frame()
        back_btn = ttk.Button(frame, text="Back", command=self.main_screen)
        frame.rowconfigure(1, weight=1)

        fig = self.filterer_instance.get_fig(plt)

        plot = FigureCanvasTkAgg(fig, frame).get_tk_widget()
        back_btn.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="E")
        plot.grid(row=1, column=0, rowspan=1, columnspan=1, sticky="N")

    def view_images(self):
        frame = self.get_frame()

        outliers = []
        similars = []
        status = ["", 0, True, 0]  # ["outliers"/"similars", image_idx, is_mosaic, non_mosaic_show_idx]

        for k, v in self.filterer_instance.failed_idxs.items():
            if v == [-1]:
                outliers.append(self.filterer_instance.data_args[k])
                continue
            similars.append([self.filterer_instance.data_args[k], [self.filterer_instance.data_args[x] for x in v]])

        # header
        back_btn = ttk.Button(frame, text="Back", command=self.main_screen)
        status_text = ttk.Label(frame, text=f"")

        similars_btn = ttk.Button(frame, text="Similars", command=lambda: mode("similars", 0, status[2], status[3]))
        outlier_btn = ttk.Button(frame, text="Outliers", command=lambda: mode("outliers", 0, status[2], status[3]))

        mosaic_checkbox = ttk.Checkbutton(frame, text="Mosaic", command=lambda: mode(status[0], status[1], not status[2], status[3]), state="disabled")
        mosaic_checkbox.state(["selected"])

        next_btn = ttk.Button(frame, text="->", command=lambda: mode(status[0], status[1] + 1, status[2], status[3]), width=2, state="disabled")
        prev_btn = ttk.Button(frame, text="<-", command=lambda: mode(status[0], status[1] - 1, status[2], status[3]), width=2, state="disabled")

        resize_slider = ttk.Scale(frame, from_=-20, to=20, orient="horizontal", length=100)
        resize_slider.set(0)

        resize_label = ttk.Label(frame, text="Resize")

        # lower part
        bottom_frame = ttk.Frame(frame)

        label1 = ttk.Label(bottom_frame, text="core img")
        label2 = ttk.Label(bottom_frame, text=f"{0}/{0}")
        next_similar_btn = ttk.Button(bottom_frame, text="->", command=lambda: mode(status[0], status[1], status[2], status[3] + 1), width=2, state="disabled")
        prev_similar_btn = ttk.Button(bottom_frame, text="<-", command=lambda: mode(status[0], status[1], status[2], status[3] - 1), width=2, state="disabled")
        image_box1 = ttk.Label(bottom_frame)
        image_box2 = ttk.Label(bottom_frame)

        def mode(img_type, index, is_mosaic, similar_idx):
            status[0] = img_type
            max_len = {"similars": len(similars),
                       "outliers": len(outliers),
                       }.get(status[0], 0)
            status[1] = min(max(0, index), max_len - 1)
            status[2] = is_mosaic
            status[3] = similar_idx

            prev_btn["state"] = "disabled" if status[1] <= 0 else "enabled"
            next_btn["state"] = "disabled" if status[1] >= max_len - 1 else "enabled"

            outlier_btn["state"] = "disabled" if not outliers else "enabled"
            similars_btn["state"] = "disabled" if not similars else "enabled"

            status_text.config(text=f"{str(status[1] + 1).zfill(len(str(max_len)))}/{max_len}")

            size_mult = 1.2 ** resize_slider.get()
            if status[0] == "similars":
                similars_btn["state"] = "disabled"
                mosaic_checkbox["state"] = "enabled"
                label1.config(text=f"Image")
                img = Image.fromarray(self.filterable_instance.get_image(similars[status[1]][0]))
                size = (max(1, int(img.size[0] * size_mult)), max(1, int(img.size[1] * size_mult)))

                label2.grid(row=0, column=4, rowspan=1, columnspan=1)
                image_box2.grid(row=1, column=3, rowspan=1, columnspan=3)

                if status[2]:  # Not mosaic
                    prev_similar_btn.grid_forget()
                    next_similar_btn.grid_forget()
                    label2.config(text=f"Removed similarities")
                    others = [Image.fromarray(self.filterable_instance.get_image(i)) for i in similars[status[1]][1]]
                    other = fit_imgs_to_grid(others)
                    status[3] = 0

                else:  # Mosaic
                    other_len = len(similars[status[1]][1])
                    status[3] = min(max(0, status[3]), other_len - 1)
                    label2.config(text=f"{str(status[3] + 1).zfill(len(str(other_len)))}/{other_len}")
                    prev_similar_btn["state"] = "disabled" if status[3] <= 0 else "enabled"
                    next_similar_btn["state"] = "disabled" if status[3] >= other_len - 1 else "enabled"
                    prev_similar_btn.grid(row=0, column=3, rowspan=1, columnspan=1, sticky="E")
                    next_similar_btn.grid(row=0, column=5, rowspan=1, columnspan=1, sticky="W")
                    other = Image.fromarray(self.filterable_instance.get_image(similars[status[1]][1][status[3]]))

                # image that is shown in tk.Label must be kept active in memory so it does not get garbage collected
                # https://stackoverflow.com/questions/16424091/why-does-tkinter-image-not-show-up-if-created-in-a-function
                self.currentimg = [ImageTk.PhotoImage(image=img.resize(size)), ImageTk.PhotoImage(image=other.resize(size))]
                image_box1.config(image=self.currentimg[0])
                image_box2.config(image=self.currentimg[1])

            elif status[0] == "outliers":
                outlier_btn["state"] = "disabled"
                mosaic_checkbox["state"] = "disabled"
                label2.grid_forget()
                image_box2.grid_forget()
                prev_similar_btn.grid_forget()
                next_similar_btn.grid_forget()
                status[3] = 0
                label1.config(text=f"Outlier")
                img = Image.fromarray(self.filterable_instance.get_image(outliers[status[1]]))
                img = img.resize((max(1, int(img.size[0] * size_mult)), max(1, int(img.size[1] * size_mult))))
                self.currentimg = ImageTk.PhotoImage(image=img)
                image_box1.config(image=self.currentimg)
            else:
                return





        resize_slider.bind("<ButtonRelease-1>", lambda x: mode(*status))
        mode(*status)

        frame.columnconfigure(6, weight=1)

        similars_btn    .grid(row=0, column=0, rowspan=1, columnspan=1)
        outlier_btn     .grid(row=1, column=0, rowspan=1, columnspan=1)

        mosaic_checkbox .grid(row=0, column=1, rowspan=1, columnspan=3, padx=(5, 5))

        prev_btn        .grid(row=1, column=1, rowspan=1, columnspan=1, sticky="E", padx=(5, 0))
        status_text     .grid(row=1, column=2, rowspan=1, columnspan=1)
        next_btn        .grid(row=1, column=3, rowspan=1, columnspan=1, sticky="W", padx=(0, 5))

        resize_label    .grid(row=0, column=5, rowspan=1, columnspan=1)
        resize_slider   .grid(row=1, column=5, rowspan=1, columnspan=1, sticky="W")

        back_btn        .grid(row=0, column=6, rowspan=2, columnspan=1, sticky="E")

        bottom_frame    .grid(row=3, column=0, rowspan=1, columnspan=7, sticky="N")
        label1          .grid(row=0, column=0, rowspan=1, columnspan=3)
        image_box1      .grid(row=1, column=0, rowspan=1, columnspan=3)

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
            self.filterer_instance = DataFilterer(model=self.filterable_instance, layer=self.chosen_layer,
                                                  device="cuda")
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


def fit_imgs_to_grid(imgs):
    if not imgs:
        return None
    mode = imgs[0].mode
    size = imgs[0].size
    grid_size = math.ceil(math.sqrt(len(imgs)))
    out = Image.new(mode, size=(0, 0))
    for i in range(grid_size):
        row = Image.new(mode, size=(0, 0))
        for j in range(grid_size):
            im = Image.new(mode, size=size)
            if i * grid_size + j < len(imgs):
                im = imgs[i * grid_size + j]
            row = concat_PIL_h(row, im)
        out = concat_PIL_v(out, row)
    return out


def concat_PIL_h(im1: PIL.Image, im2: PIL.Image):
    dst = Image.new(im1.mode, (im1.width + im2.width, max(im1.height, im2.height)))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def concat_PIL_v(im1: PIL.Image, im2: PIL.Image):
    dst = Image.new(im1.mode, (max(im1.width, im2.width), im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


if __name__ == "__main__":
    app = app()
    app.master.attributes('-type', 'dialog')
    app.master.mainloop()

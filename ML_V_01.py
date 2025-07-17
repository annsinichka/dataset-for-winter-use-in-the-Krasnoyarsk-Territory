import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import os
from scipy import stats

class RegressionApp:
    LANGS = {
        'ru': {
            'title': 'Анализ регрессии',
            'select_file': 'Выбрать файл',
            'file_not_selected': 'Файл не выбран',
            'available_columns': 'Доступные колонки:',
            'target_variable': 'Целевая переменная:',
            'show_corr_matrix': 'Показать корреляции',
            'run_regression': 'Выполнить регрессию',
            'reset_selection': 'Сбросить выбор',
            'train_size': 'Размер обучающей выборки (%)',
            'test_size': 'Размер тестовой выборки (%)',
            'max_rows_label': 'Макс. строк теста (0=все, до 10%):',
            'encoding_select': 'Кодировка CSV:',
            'corr_method': 'Метод корреляции:',
            'warning_load_data': 'Сначала загрузите данные',
            'warning_no_numeric_cols': 'Нет числовых колонок для анализа',
            'warning_select_feature': 'Выберите хотя бы один признак',
            'warning_select_target': 'Выберите целевую переменную',
            'warning_target_in_features': 'Целевая переменная не может быть среди признаков',
            'warning_invalid_train_size': 'Введите корректный процент обучения (0-100)',
            'warning_invalid_max_rows': 'Введите корректное макс. число строк (>=0)',
            'error_no_data_after_nan': 'Нет данных для анализа после удаления пропущенных значений',
            'error_loading_file': 'Не удалось загрузить файл',
            'error_regression': 'Произошла ошибка при выполнении регрессии',
            'corr_matrix_title': 'Матрица корреляций',
            'feature_importance': 'Важность признаков',
            'feature_corr': 'Корреляции признаков',
            'multiple_corr': 'Множественная корреляция',
            'language_select': 'Язык / Language:',
            'results_title': 'Результаты регрессии',
            'model_col': 'Модель',
            'mse_col': 'MSE',
            'rmse_col': 'RMSE',
            'r2_col': 'R²',
            'r_col': 'R',
            'actual_values': 'Фактические значения',
            'predicted_values': 'Предсказанные значения',
            'train_size_label': 'Размер обучающей выборки (%):',
            'save_plot': 'Сохранить график',
            'save_title': 'Сохранить график как PNG',
            'save_success': 'График успешно сохранен',
            'save_error': 'Ошибка при сохранении графика',
            'no_feature_importance': 'Важность признаков недоступна для этой модели',
            'selected_corr_matrix': 'Матрица корреляций (выбранные столбцы)',
            'multiple_corr_value': 'Множественная корреляция (R):',
        },
        'en': {
            'title': 'Regression Analysis',
            'select_file': 'Select File',
            'file_not_selected': 'File not selected',
            'available_columns': 'Available Columns:',
            'target_variable': 'Target Variable:',
            'show_corr_matrix': 'Show Correlations',
            'run_regression': 'Run Regression',
            'reset_selection': 'Reset Selection',
            'train_size': 'Training Set Size (%)',
            'test_size': 'Test Set Size (%)',
            'max_rows_label': 'Max test rows (0=all, up to 10%):',
            'encoding_select': 'CSV Encoding:',
            'corr_method': 'Correlation method:',
            'warning_load_data': 'Please load data first',
            'warning_no_numeric_cols': 'No numeric columns for analysis',
            'warning_select_feature': 'Select at least one feature',
            'warning_select_target': 'Select target variable',
            'warning_target_in_features': 'Target variable cannot be among features',
            'warning_invalid_train_size': 'Enter valid training percent (0-100)',
            'warning_invalid_max_rows': 'Enter valid max rows (>=0)',
            'error_no_data_after_nan': 'No data available after removing missing values',
            'error_loading_file': 'Failed to load file',
            'error_regression': 'An error occurred during regression',
            'corr_matrix_title': 'Correlation Matrix',
            'feature_importance': 'Feature Importance',
            'feature_corr': 'Feature Correlations',
            'multiple_corr': 'Multiple Correlation',
            'language_select': 'Язык / Language:',
            'results_title': 'Regression Results',
            'model_col': 'Model',
            'mse_col': 'MSE',
            'rmse_col': 'RMSE',
            'r2_col': 'R²',
            'r_col': 'R',
            'actual_values': 'Actual values',
            'predicted_values': 'Predicted values',
            'train_size_label': 'Training set size (%):',
            'save_plot': 'Save Plot',
            'save_title': 'Save Plot as PNG',
            'save_success': 'Plot saved successfully',
            'save_error': 'Error saving plot',
            'no_feature_importance': 'Feature importance not available for this model',
            'selected_corr_matrix': 'Correlation Matrix (Selected Columns)',
            'multiple_corr_value': 'Multiple Correlation (R):',
        }
    }

    def __init__(self, root):
        self.root = root
        self.lang = 'ru'
        self.strings = self.LANGS[self.lang]
        self.root.title(self.strings['title'])
        self.root.geometry("1200x900")
        self.root.minsize(1000, 700)
        
        self.data = None
        self.features = []
        self.target = None
        self.selected_features = []
        self.encoding = 'utf-8-sig'
        
        self.labels = {}
        self.buttons = {}
        
        self.create_widgets()
    
    def create_widgets(self):
        # Language selection
        lang_frame = tk.Frame(self.root)
        lang_frame.pack(fill=tk.X, padx=10, pady=5, anchor='e')
        
        self.labels['language_select'] = tk.Label(lang_frame, text=self.strings['language_select'])
        self.labels['language_select'].pack(side=tk.LEFT)
        
        self.lang_combo = ttk.Combobox(lang_frame, state='readonly', values=['Русский', 'English'], width=10)
        self.lang_combo.current(0 if self.lang == 'ru' else 1)
        self.lang_combo.pack(side=tk.LEFT, padx=5)
        self.lang_combo.bind("<<ComboboxSelected>>", self.change_language)
        
        # Frame for file selection & encoding
        file_frame = tk.Frame(self.root)
        file_frame.pack(pady=10, fill=tk.X, padx=10)
        
        self.buttons['select_file'] = tk.Button(file_frame, text=self.strings['select_file'], command=self.load_file)
        self.buttons['select_file'].pack(side=tk.LEFT, padx=5)
        
        self.labels['file_status'] = tk.Label(file_frame, text=self.strings['file_not_selected'])
        self.labels['file_status'].pack(side=tk.LEFT, padx=5)
        
        self.labels['encoding_label'] = tk.Label(file_frame, text=self.strings['encoding_select'])
        self.labels['encoding_label'].pack(side=tk.LEFT, padx=(20,5))
        
        self.encoding_combo = ttk.Combobox(file_frame, state='readonly', values=['utf-8', 'utf-8-sig', 'cp1251'], width=10)
        self.encoding_combo.current(1)
        self.encoding_combo.pack(side=tk.LEFT)
        
        # Frame for data display
        data_frame = tk.LabelFrame(self.root, text="Предпросмотр данных")
        data_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview for data preview
        tree_scroll = ttk.Scrollbar(data_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree = ttk.Treeview(data_frame, yscrollcommand=tree_scroll.set)
        self.tree.pack(fill=tk.BOTH, expand=True)
        tree_scroll.config(command=self.tree.yview)
        
        # Frame for feature selection and controls
        selection_frame = tk.LabelFrame(self.root, text="Выбор переменных")
        selection_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Feature selection listbox
        self.labels['features_label'] = tk.Label(selection_frame, text=self.strings['available_columns'])
        self.labels['features_label'].grid(row=0, column=0, sticky='w', padx=5, pady=2)
        
        feature_list_scroll = ttk.Scrollbar(selection_frame)
        self.feature_listbox = tk.Listbox(selection_frame, selectmode=tk.MULTIPLE, height=6, yscrollcommand=feature_list_scroll.set)
        self.feature_listbox.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        feature_list_scroll.config(command=self.feature_listbox.yview)
        feature_list_scroll.grid(row=1, column=1, sticky='ns')
        
        # Target selection combobox
        self.labels['target_label'] = tk.Label(selection_frame, text=self.strings['target_variable'])
        self.labels['target_label'].grid(row=0, column=2, sticky='w', padx=5, pady=2)
        
        self.target_combobox = ttk.Combobox(selection_frame, state="readonly", width=30)
        self.target_combobox.grid(row=1, column=2, sticky='ew', padx=5, pady=5)
        
        # Correlation method
        self.labels['corr_method_label'] = tk.Label(selection_frame, text=self.strings['corr_method'])
        self.labels['corr_method_label'].grid(row=0, column=3, sticky='w', padx=5, pady=2)
        
        self.corr_method_combobox = ttk.Combobox(selection_frame, state="readonly", 
                                                values=["pearson", "kendall", "spearman"], width=10)
        self.corr_method_combobox.set("pearson")
        self.corr_method_combobox.grid(row=1, column=3, sticky='ew', padx=5, pady=5)
        
        # Training controls frame
        train_control_frame = tk.LabelFrame(self.root, text="Параметры обучения")
        train_control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Training size controls
        train_size_frame = tk.Frame(train_control_frame)
        train_size_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=5)
        
        self.labels['train_size_label'] = tk.Label(train_size_frame, text=self.strings['train_size_label'])
        self.labels['train_size_label'].pack(anchor=tk.W)
        
        self.train_size_var = tk.StringVar(value="30")  # 30% обучающая выборка
        self.train_size_entry = tk.Entry(train_size_frame, width=5, textvariable=self.train_size_var)
        self.train_size_entry.pack(anchor='w')
        
        # Max rows controls
        max_rows_frame = tk.Frame(train_control_frame)
        max_rows_frame.pack(side=tk.LEFT, fill=tk.X, padx=20)
        
        self.labels['max_rows_label'] = tk.Label(max_rows_frame, text=self.strings['max_rows_label'])
        self.labels['max_rows_label'].pack(anchor=tk.W)
        
        self.max_rows_var = tk.StringVar(value="0")
        self.max_rows_entry = tk.Entry(max_rows_frame, width=10, textvariable=self.max_rows_var)
        self.max_rows_entry.pack(anchor='w')
        
        # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.buttons['show_corr'] = tk.Button(button_frame, text=self.strings['show_corr_matrix'], 
                                            command=self.show_correlation_matrix, width=20)
        self.buttons['show_corr'].pack(side=tk.LEFT, padx=5)
        
        self.buttons['run_regression'] = tk.Button(button_frame, text=self.strings['run_regression'], 
                                                 command=self.run_regression, width=20)
        self.buttons['run_regression'].pack(side=tk.LEFT, padx=5)
        
        self.buttons['reset'] = tk.Button(button_frame, text=self.strings['reset_selection'], 
                                        command=self.reset_selection, width=20)
        self.buttons['reset'].pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        selection_frame.columnconfigure(0, weight=3)
        selection_frame.columnconfigure(2, weight=2)
        selection_frame.columnconfigure(3, weight=1)
    
    def change_language(self, event=None):
        sel = self.lang_combo.get()
        self.lang = 'ru' if sel == 'Русский' else 'en'
        self.strings = self.LANGS[self.lang]
        self.update_ui_texts()
    
    def update_ui_texts(self):
        self.root.title(self.strings['title'])
        self.buttons['select_file'].config(text=self.strings['select_file'])
        self.buttons['show_corr'].config(text=self.strings['show_corr_matrix'])
        self.buttons['run_regression'].config(text=self.strings['run_regression'])
        self.buttons['reset'].config(text=self.strings['reset_selection'])
        self.labels['language_select'].config(text=self.strings['language_select'])
        self.labels['file_status'].config(text=self.strings['file_not_selected'] if self.data is None else self.labels['file_status'].cget("text"))
        self.labels['encoding_label'].config(text=self.strings['encoding_select'])
        self.labels['features_label'].config(text=self.strings['available_columns'])
        self.labels['target_label'].config(text=self.strings['target_variable'])
        self.labels['corr_method_label'].config(text=self.strings['corr_method'])
        self.labels['train_size_label'].config(text=self.strings['train_size_label'])
        self.labels['max_rows_label'].config(text=self.strings['max_rows_label'])
    
    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if not filepath:
            return
        enc = self.encoding_combo.get()
        try:
            self.data = pd.read_csv(filepath, encoding=enc, sep=';')
            
            # Create has_name columns only for string columns that don't already have such a column
            for col in self.data.columns:
                if self.data[col].dtype == object and not col.endswith('_has_name'):
                    has_name_col = col + '_has_name'
                    if has_name_col not in self.data.columns:
                        self.data[has_name_col] = self.data[col].apply(
                            lambda x: 1 if isinstance(x, str) and x.strip() != '' else 0)
            
            # Convert comma strings to float
            for col in self.data.columns:
                if self.data[col].dtype == object and not col.endswith('_has_name'):
                    try:
                        self.data[col] = self.data[col].str.replace(',', '.').astype(float)
                    except Exception:
                        pass
            
            self.labels['file_status'].config(text=filepath.split('/')[-1])
            self.update_feature_lists()
            self.show_data_preview()
        except Exception as e:
            messagebox.showerror(self.strings['error_loading_file'], str(e))
            self.data = None
            self.labels['file_status'].config(text=self.strings['file_not_selected'])
            self.feature_listbox.delete(0, tk.END)
            self.target_combobox['values'] = []
            self.tree.delete(*self.tree.get_children())
    
    def update_feature_lists(self):
        if self.data is None:
            self.feature_listbox.delete(0, tk.END)
            self.target_combobox['values'] = []
            return
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            messagebox.showwarning(self.strings['title'], self.strings['warning_no_numeric_cols'])
            return
        
        self.feature_listbox.delete(0, tk.END)
        for col in numeric_cols:
            self.feature_listbox.insert(tk.END, col)
        
        self.target_combobox['values'] = numeric_cols
        if numeric_cols:
            self.target_combobox.current(0)
    
    def show_data_preview(self):
        if self.data is None:
            return
            
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(self.data.columns)
        self.tree["show"] = "headings"
        
        # Configure columns
        for col in self.data.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=80, anchor='center', minwidth=50)
        
        # Insert data
        for i, row in self.data.head(100).iterrows():
            vals = list(row.values)
            vals = ["" if pd.isna(x) else f"{x:.3f}" if isinstance(x, float) else str(x) for x in vals]
            self.tree.insert("", "end", values=vals)
    
    def reset_selection(self):
        self.feature_listbox.selection_clear(0, tk.END)
        if self.target_combobox['values']:
            self.target_combobox.current(0)
        self.train_size_var.set("30")
        self.max_rows_var.set("0")
    
    def show_correlation_matrix(self):
        if self.data is None:
            messagebox.showwarning(self.strings['title'], self.strings['warning_load_data'])
            return
        
        # Get selected features and target
        selected_features = [self.feature_listbox.get(i) for i in self.feature_listbox.curselection()]
        target = self.target_combobox.get()
        
        method = self.corr_method_combobox.get()
        
        # Show full correlation matrix if no features are selected
        if not selected_features or not target:
            numeric_data = self.data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                messagebox.showwarning(self.strings['title'], self.strings['warning_no_numeric_cols'])
                return
            
            try:
                corr = numeric_data.corr(method=method)
                title = self.strings['corr_matrix_title']
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка расчета корреляции: {str(e)}")
                return
        else:
            # Show correlation matrix for selected columns
            cols = selected_features + [target]
            numeric_data = self.data[cols].select_dtypes(include=[np.number])
            if numeric_data.empty:
                messagebox.showwarning(self.strings['title'], self.strings['warning_no_numeric_cols'])
                return
            
            try:
                corr = numeric_data.corr(method=method)
                title = self.strings['selected_corr_matrix']
                
                # Calculate multiple correlation coefficient
                try:
                    y = numeric_data[target]
                    X = numeric_data[selected_features]
                    valid_rows = ~X.isnull().any(axis=1) & ~y.isnull()
                    X = X[valid_rows]
                    y = y[valid_rows]
                    
                    if len(X) > 0:
                        # Calculate R (multiple correlation coefficient)
                        model = LinearRegression()
                        model.fit(X, y)
                        y_pred = model.predict(X)
                        r2 = r2_score(y, y_pred)
                        r_value = np.sqrt(r2) if r2 >= 0 else float('nan')
                        
                        # Add multiple correlation to the matrix
                        multiple_corr_row = pd.Series([r_value] * len(corr.columns), 
                                                    index=corr.columns, 
                                                    name=self.strings['multiple_corr_value'])
                        corr = pd.concat([corr, pd.DataFrame([multiple_corr_row])])
                    else:
                        messagebox.showwarning("Предупреждение", "Недостаточно данных для расчета множественной корреляции")
                except Exception as e:
                    messagebox.showwarning("Предупреждение", f"Не удалось рассчитать множественную корреляцию: {str(e)}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка расчета корреляции: {str(e)}")
                return
        
        # Create a new window for correlation matrix
        corr_window = tk.Toplevel(self.root)
        corr_window.title(title)
        corr_window.geometry("1200x800")
        
        # Main container with PanedWindow for resizable layout
        main_pane = tk.PanedWindow(corr_window, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)
        
        # Left frame for the plot
        plot_frame = tk.Frame(main_pane)
        main_pane.add(plot_frame, width=1000)  # Fixed width for plot area
        
        # Right frame for the toolbar
        toolbar_frame = tk.Frame(main_pane)
        main_pane.add(toolbar_frame, width=200)  # Fixed width for toolbar
        
        # Create the figure and axis with constrained layout
        fig = plt.Figure(figsize=(10, 8), dpi=100, constrained_layout=True)
        ax = fig.add_subplot(111)
        
        # Generate the heatmap with fixed color scale (-1 to 1)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', 
                   center=0, vmin=-1, vmax=1, ax=ax,
                   cbar_kws={"shrink": 0.8, "orientation": "vertical"})
        
        ax.set_title(f"{title} ({method})", pad=20)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create vertical toolbar on the right side
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        # Configure toolbar buttons to be vertical
        for btn in toolbar.winfo_children():
            btn.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)
        
        # Add save button to toolbar
        save_button = tk.Button(
            toolbar_frame, 
            text=self.strings['save_plot'], 
            command=lambda: self.save_plot(fig, title),
            width=15
        )
        save_button.pack(side=tk.TOP, padx=5, pady=5)
        
        # Configure weights for resizing
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        toolbar_frame.columnconfigure(0, weight=1)
    
    def save_plot(self, fig, title):
        filepath = filedialog.asksaveasfilename(
            title=self.strings['save_title'],
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            defaultextension=".png"
        )
        
        if not filepath:
            return
        
        try:
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            messagebox.showinfo(self.strings['save_success'], 
                               f"{self.strings['save_success']}\n{os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror(self.strings['save_error'], f"{self.strings['save_error']}: {str(e)}")
    
    def run_regression(self):
        if self.data is None:
            messagebox.showwarning(self.strings['title'], self.strings['warning_load_data'])
            return
        
        self.selected_features = [self.feature_listbox.get(i) for i in self.feature_listbox.curselection()]
        self.target = self.target_combobox.get()
        
        if not self.selected_features:
            messagebox.showwarning(self.strings['title'], self.strings['warning_select_feature'])
            return
        
        if not self.target:
            messagebox.showwarning(self.strings['title'], self.strings['warning_select_target'])
            return
        
        if self.target in self.selected_features:
            messagebox.showwarning(self.strings['title'], self.strings['warning_target_in_features'])
            return
        
        # Validate training size
        try:
            train_size_percent = float(self.train_size_var.get())
            if not (0 < train_size_percent < 100):
                raise ValueError()
            test_size = (100 - train_size_percent) / 100.0
        except:
            messagebox.showerror("Ошибка", self.strings['warning_invalid_train_size'])
            return
        
        # Validate max rows
        try:
            max_rows = int(self.max_rows_var.get())
            if max_rows < 0:
                raise ValueError()
        except:
            messagebox.showerror("Ошибка", self.strings['warning_invalid_max_rows'])
            return
        
        try:
            X = self.data[self.selected_features]
            y = self.data[self.target]

            # Remove rows with missing values
            valid_rows = ~X.isnull().any(axis=1) & ~y.isnull()
            X = X[valid_rows]
            y = y[valid_rows]

            if len(X) == 0:
                messagebox.showerror(self.strings['title'], self.strings['error_no_data_after_nan'])
                return

            # Split data (training set = 30%)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Limit test set size
            if max_rows > 0:
                max_allowed = max(1, int(len(X_test) * 0.1))
                if max_rows > max_allowed:
                    max_rows = max_allowed
                X_test = X_test.iloc[:max_rows]
                y_test = y_test.iloc[:max_rows]

            # Preprocessing pipeline
            preprocessor = make_pipeline(
                SimpleImputer(strategy='median'), 
                StandardScaler()
            )
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            # Regression models
            models = [
                ('Линейная регрессия', LinearRegression()),
                ('Гребневая регрессия (Ridge)', Ridge(alpha=1.0)),
                ('Лассо регрессия (Lasso)', Lasso(alpha=0.1)),
                ('Дерево решений', DecisionTreeRegressor(max_depth=5, random_state=42)),
                ('Случайный лес', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('Градиентный бустинг', GradientBoostingRegressor(n_estimators=100, random_state=42)),
                ('HistGradientBoosting', HistGradientBoostingRegressor(random_state=42))
            ]

            # Create results window
            result_window = tk.Toplevel(self.root)
            result_window.title(self.strings['results_title'])
            result_window.geometry("1200x900")
            
            notebook = ttk.Notebook(result_window)
            notebook.pack(fill=tk.BOTH, expand=True)

            # Results table tab
            results_frame = ttk.Frame(notebook)
            notebook.add(results_frame, text=self.strings['results_title'])
            
            # Create results table
            results_tree = ttk.Treeview(
                results_frame, 
                columns=("Model", "MSE", "RMSE", "R²", "R"), 
                show="headings",
                height=20
            )
            results_tree.heading("Model", text=self.strings['model_col'])
            results_tree.heading("MSE", text=self.strings['mse_col'])
            results_tree.heading("RMSE", text=self.strings['rmse_col'])
            results_tree.heading("R²", text=self.strings['r2_col'])
            results_tree.heading("R", text=self.strings['r_col'])
            
            # Set column widths
            results_tree.column("Model", width=250, anchor='w')
            results_tree.column("MSE", width=150, anchor='center')
            results_tree.column("RMSE", width=150, anchor='center')
            results_tree.column("R²", width=150, anchor='center')
            results_tree.column("R", width=150, anchor='center')
            
            results_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=results_tree.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            results_tree.configure(yscrollcommand=scrollbar.set)

            # Train models and create plots
            for name, model in models:
                try:
                    model.fit(X_train_processed, y_train)
                    y_pred = model.predict(X_test_processed)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Calculate multiple correlation R
                    r_value = np.sqrt(r2) if r2 >= 0 else float('nan')
                    
                    # Add results to table
                    results_tree.insert("", tk.END, values=(
                        name, 
                        f"{mse:.4f}", 
                        f"{rmse:.4f}", 
                        f"{r2:.4f}",
                        f"{r_value:.4f}"
                    ))

                    # Create model tab with plots
                    model_frame = ttk.Frame(notebook)
                    notebook.add(model_frame, text=name)
                    
                    # Create container for plots
                    plot_container = tk.PanedWindow(model_frame, orient=tk.VERTICAL)
                    plot_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                    
                    # Frame for actual vs predicted plot
                    actual_pred_frame = ttk.LabelFrame(plot_container, text="Фактические vs Предсказанные значения")
                    plot_container.add(actual_pred_frame)
                    
                    # Create actual vs predicted plot
                    fig1 = Figure(figsize=(10, 4), dpi=100)
                    ax1 = fig1.add_subplot(111)
                    ax1.scatter(y_test, y_pred, alpha=0.5)
                    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                    ax1.set_title(f'{name} - R²={r2:.3f}')
                    ax1.set_xlabel(self.strings['actual_values'])
                    ax1.set_ylabel(self.strings['predicted_values'])
                    ax1.grid(True)
                    
                    canvas1 = FigureCanvasTkAgg(fig1, master=actual_pred_frame)
                    canvas1.draw()
                    canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                    
                    # Add toolbar for the first plot
                    toolbar_frame1 = tk.Frame(actual_pred_frame)
                    toolbar_frame1.pack(fill=tk.X)
                    toolbar1 = NavigationToolbar2Tk(canvas1, toolbar_frame1)
                    toolbar1.update()
                    
                    # Add save button
                    save_button1 = tk.Button(
                        toolbar_frame1, 
                        text=self.strings['save_plot'], 
                        command=lambda fig=fig1, title=f"{name}_actual_vs_predicted": self.save_plot(fig, title)
                    )
                    save_button1.pack(side=tk.RIGHT, padx=5, pady=2)
                    
                    # Create feature importance plot if available
                    feature_frame = ttk.LabelFrame(plot_container, text=self.strings['feature_importance'])
                    plot_container.add(feature_frame)
                    
                    fig2 = Figure(figsize=(10, 4), dpi=100)
                    ax2 = fig2.add_subplot(111)
                    
                    if hasattr(model, 'feature_importances_'):
                        # For tree-based models
                        importances = model.feature_importances_
                        indices = np.argsort(importances)[::-1]
                        
                        # Create horizontal bar plot
                        ax2.barh(range(len(indices)), importances[indices], align='center')
                        ax2.set_yticks(range(len(indices)))
                        ax2.set_yticklabels([self.selected_features[i] for i in indices])
                        ax2.set_title(self.strings['feature_importance'])
                        ax2.set_xlabel('Важность')
                        ax2.grid(True, axis='x')
                    
                    elif hasattr(model, 'coef_'):
                        # For linear models
                        coef = model.coef_
                        indices = np.argsort(np.abs(coef))[::-1]
                        
                        # Create horizontal bar plot
                        ax2.barh(range(len(indices)), coef[indices], align='center')
                        ax2.set_yticks(range(len(indices)))
                        ax2.set_yticklabels([self.selected_features[i] for i in indices])
                        ax2.set_title(self.strings['feature_corr'])
                        ax2.set_xlabel('Коэффициент')
                        ax2.grid(True, axis='x')
                    
                    else:
                        # No feature importance available
                        ax2.text(0.5, 0.5, self.strings['no_feature_importance'], 
                                ha='center', va='center', fontsize=12)
                    
                    canvas2 = FigureCanvasTkAgg(fig2, master=feature_frame)
                    canvas2.draw()
                    canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                    
                    # Add toolbar for the second plot
                    toolbar_frame2 = tk.Frame(feature_frame)
                    toolbar_frame2.pack(fill=tk.X)
                    toolbar2 = NavigationToolbar2Tk(canvas2, toolbar_frame2)
                    toolbar2.update()
                    
                    # Add save button
                    save_button2 = tk.Button(
                        toolbar_frame2, 
                        text=self.strings['save_plot'], 
                        command=lambda fig=fig2, title=f"{name}_feature_importance": self.save_plot(fig, title)
                    )
                    save_button2.pack(side=tk.RIGHT, padx=5, pady=2)
                    
                except Exception as e:
                    results_tree.insert("", tk.END, values=(
                        name, 
                        "Ошибка", 
                        str(e), 
                        "",
                        ""
                    ))

        except Exception as e:
            messagebox.showerror(
                self.strings['error_regression'], 
                f"{self.strings['error_regression']}: {str(e)}"
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = RegressionApp(root)
    root.mainloop()
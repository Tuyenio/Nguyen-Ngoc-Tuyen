import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Button, Label, filedialog, Frame, Toplevel, messagebox
from PIL import Image, ImageTk

# Hàm hiển thị ảnh bằng matplotlib
def display_image(image, title, cmap='gray'):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')  # Tắt trục hiển thị
    plt.show()

# Hàm dò biên bằng toán tử Sobel
def sobel_edge_detection(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
    return sobel_combined

# Hàm dò biên bằng Laplace Gaussian
def laplace_gaussian_edge_detection(image):
    laplacian_gaussian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian_gaussian = cv2.normalize(laplacian_gaussian, None, 0, 255, cv2.NORM_MINMAX)
    return laplacian_gaussian

# Hàm xử lý khi chọn ảnh
def open_file():
    global img_path, img_display
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    
    if not file_path:
        return
    
    img_path = file_path
    
    # Đọc và hiển thị ảnh đã chọn
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        messagebox.showerror("Lỗi", "Không thể đọc ảnh. Hãy kiểm tra lại.")
        return
    
    # Hiển thị ảnh đã chọn trong khung
    img_display = cv2.cvtColor(cv2.resize(image, (300, 300)), cv2.COLOR_GRAY2RGB)
    img_display = ImageTk.PhotoImage(image=Image.fromarray(img_display))
    img_label.config(image=img_display)
    img_label.image = img_display

# Hàm xử lý dò biên
def process_image():
    if img_path is None:
        messagebox.showwarning("Cảnh báo", "Vui lòng chọn một ảnh trước khi xử lý.")
        return
    
    # Đọc ảnh và thực hiện xử lý
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    sobel_result = sobel_edge_detection(image)
    laplacian_result = laplace_gaussian_edge_detection(image)
    
    # Hiển thị kết quả
    display_image(sobel_result, 'Sobel Combined')
    display_image(laplacian_result, 'Laplace Gaussian')

# Hàm khởi tạo giao diện
def create_gui():
    global img_label, img_path
    
    # Khởi tạo cửa sổ chính
    window = Tk()
    window.title("Dò Biên Ảnh Cao Cấp")
    window.geometry("700x600")
    window.config(bg="#ececec")  # Màu nền nhạt cao cấp

    # Tiêu đề
    title_label = Label(window, text="Dò Biên Ảnh - Sobel và Laplace Gaussian", font=("Helvetica", 24, "bold"), bg="#ececec", fg="#333333")
    title_label.pack(pady=20)

    # Khung chứa ảnh và khung kết quả
    main_frame = Frame(window, bg="#ececec", padx=10, pady=10)
    main_frame.pack(pady=10)
    
    # Khung hiển thị ảnh
    img_frame = Frame(main_frame, bg="#ffffff", bd=3, relief="ridge")
    img_frame.grid(row=0, column=0, padx=20, pady=10)

    # Nhãn để hiển thị ảnh
    img_label = Label(img_frame)
    img_label.pack()

    # Khung chứa các nút điều khiển
    control_frame = Frame(main_frame, bg="#ececec")
    control_frame.grid(row=0, column=1, padx=20, pady=10)
    
    # Nút chọn ảnh
    select_button = Button(control_frame, text="Chọn Ảnh", font=("Helvetica", 14), bg="#0052cc", fg="#ffffff",
                           command=open_file, padx=20, pady=10)
    select_button.grid(row=0, column=0, pady=10)

    # Nút xử lý dò biên
    process_button = Button(control_frame, text="Xử Lý Ảnh", font=("Helvetica", 14), bg="#28a745", fg="#ffffff",
                            command=process_image, padx=20, pady=10)
    process_button.grid(row=1, column=0, pady=10)

    # Khung thông tin kết quả
    result_frame = Frame(window, bg="#ffffff", bd=3, relief="groove")
    result_frame.pack(padx=20, pady=20)

    # Nhãn thông tin
    result_label = Label(result_frame, text="Kết quả xử lý sẽ hiển thị tại đây.", font=("Helvetica", 14), bg="#ffffff", fg="#333333")
    result_label.pack(pady=10, padx=10)
    
    # Khởi tạo đường dẫn ảnh
    img_path = None

    # Chạy vòng lặp giao diện
    window.mainloop()

# Gọi hàm khởi tạo GUI
create_gui()

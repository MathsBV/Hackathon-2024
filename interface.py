import cv2
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture
from kivy.uix.label import Label
from liveface import ReconhecimentoFacial

class CameraApp(App):
    def build(self):
        self.layout_principal = BoxLayout(orientation='vertical')

        # BoxLayout for camera with gray border
        self.layout_camera = BoxLayout()
        self.layout_camera.size_hint = (None, None)
        self.layout_camera.width = 640  # Camera width
        self.layout_camera.height = 480  # Camera height
        self.layout_camera.padding = 10
        self.layout_camera.border = (2, 2, 2, 2)
        self.layout_camera.canvas.add(Color(0.5, 0.5, 0.5, 1))  # Gray border color
        self.layout_principal.size_hint = (0.7, 0.8) 

        self.camera = cv2.VideoCapture(0)
        Clock.schedule_interval(self.atualizar_frame, 1.0 / 30.0)  # Update camera frame every 1/30 second
        self.img_camera = Image()
        self.layout_camera.add_widget(self.img_camera)
        self.layout_principal.add_widget(self.layout_camera)

        # BoxLayout for buttons (centered horizontally with camera)
        self.layout_botoes = BoxLayout(orientation='horizontal', size_hint=(None, None), width=640, height=100)
        self.layout_botoes.pos_hint = {'center_x': 0.5}  # Center buttons horizontally with camera

        self.layout_botoes.spacing = 50  # Spacing between buttons

        self.btn_cadastrar_rosto = Button(text="Cadastrar Rosto", size_hint=(None, None), width=250, height=100)
        self.btn_cadastrar_rosto.bind(on_press=self.cadastrar_rosto)
        self.btn_login_rosto = Button(text="Reconhecimento Facial", size_hint=(None, None), width=250, height=100)
        self.btn_login_rosto.bind(on_press=self.reconhecer_rosto)
        self.layout_botoes.add_widget(self.btn_cadastrar_rosto)
        self.layout_botoes.add_widget(self.btn_login_rosto)

        self.layout_principal.add_widget(self.layout_botoes)

        # Message label above buttons
        self.message_label = Label(text="Escolha a sua opção: ", size_hint=(None, None), size=(200, 100))
        self.layout_principal.add_widget(self.message_label)
        self.message_label.pos_hint = {'center_x': 0.5}  # Center label horizontally
        return self.layout_principal
    
    def update_message(self, message):
        self.message_label.text = message
    def atualizar_frame(self, dt):
        ret, frame = self.camera.read()
        if ret:
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img_camera.texture = texture

    def cadastrar_rosto(self, instance):
        print("Cadastrando rosto...")
        ret, frame = self.camera.read()
        if ret:
            # Salva o frame como uma imagem
            cv2.imwrite("rosto.jpeg", frame)
            print("Rosto cadastrado com sucesso!")
        else:
            print("Erro ao cadastrar rosto")

    def reconhecer_rosto(self, instance):
        reconhecimento = ReconhecimentoFacial("rosto.jpeg", "shape_predictor_68_face_landmarks.dat", "dlib_face_recognition_resnet_model_v1.dat")
        print("Botão de reconhecimento facial pressionado")
        while True:
            ret, frame = self.camera.read()
            if reconhecimento.reconhecer_rosto(frame):
                cv2.putText(frame, "Face reconhecida!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Reconhecimento Facial", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.camera.release()

if __name__ == "__main__":
    CameraApp().run()

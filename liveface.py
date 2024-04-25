import cv2
import dlib
import numpy as np

class ReconhecimentoFacial:
    def __init__(self, referencia_path, predictor_path, facerec_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.facerec = dlib.face_recognition_model_v1(facerec_path)
        self.descritores_referencia = self._extrair_descritores_referencia(referencia_path)

    def _extrair_descritores_referencia(self, referencia_path):
        imagem_referencia = cv2.imread(referencia_path)
        imagem_referencia_gray = cv2.cvtColor(imagem_referencia, cv2.COLOR_BGR2GRAY)
        caixas_faces = self.detector(imagem_referencia_gray)
        return self._extrair_descritores_faciais(imagem_referencia, caixas_faces)

    def _extrair_descritores_faciais(self, imagem, caixas_faces):
        descritores = []
        for caixa_face in caixas_faces:
            pontos_chave = self.predictor(imagem, caixa_face)
            descritor = self.facerec.compute_face_descriptor(imagem, pontos_chave)
            descritores.append(descritor)
        return descritores

    def reconhecer_rosto(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        caixas_faces = self.detector(frame_gray)
        descritores_frame = self._extrair_descritores_faciais(frame, caixas_faces)
        for descritor_frame in descritores_frame:
            for descritor_referencia in self.descritores_referencia:
                distancia = np.linalg.norm(np.array(descritor_frame) - np.array(descritor_referencia))
                if distancia < 0.5:
                    return True
        return False

if __name__ == "__main__":
    reconhecimento = ReconhecimentoFacial("rosto.png", "shape_predictor_68_face_landmarks.dat", "dlib_face_recognition_resnet_model_v1.dat")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if reconhecimento.reconhecer_rosto(frame):
            cv2.putText(frame, "Face reconhecida!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Reconhecimento Facial", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

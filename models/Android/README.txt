ʹ��Android Studio��Ŀ¼��ַ\substation\models\Android\tensorflow-yolo\tensorflow\examples�µ�Android���̡�

Android Studio����ʵ��tiny-YOLOv3Ŀ���⹤����Android�ƶ��˵�Ӧ�ã�ģ���ļ�Ҫ��tensorflow��pb�ļ�������TensorflowInferenceInterface
�ӿ�ʵ��pbģ���ļ����ƶϹ��̣���������ӿ��ṩ��fetch�����ٽ�ϵݹ�ѭ����ѯ����һ����confidence�ʹ洢��priorityQueue�У�tiny-yolov3����
��������Ұ�ֱ�Ϊ13*13��26*26��·����ֱ�洢������priorityQueue�У�������ȡ����queue��confidence��top n����

TODO:
1. Ѱ�Һ��ʵ�anchorsֵ
2. ������Ŀ���̵Ĳ�ͬ��k-means�ķ����õ����ʵ�anchors��ֵ
3. YOLODetector���漰offset�ļ����Լ���NUM_BOXES_PER_BLOCK�����Ĺ�ϵ��δŪ�壺offset�������Ϊoutput tensor������������ȡ��λ������ȵ�output��Ϣ
4.��Ҫ����offset��output tensor�Ǹ��ط���output tensorĿǰȡ�Ĵ�С����Ӧ����13* 13* (2+ 5) * 6/2
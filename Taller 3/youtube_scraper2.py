import pandas as pd
from googleapiclient.discovery import build
from time import sleep
import traceback

# Script para extraer comentarios de YouTube con likes y adaptarlo al formato de la app Streamlit

def get_comments(api_key, video_id):
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.commentThreads().list(
        part="snippet,replies",
        videoId=video_id,
        textFormat="plainText",
        maxResults=100
    )

    comments, replies, user_names, dates, likes = [], [], [], [], []
    df = pd.DataFrame()

    while request:
        try:
            response = request.execute()
            for item in response['items']:
                # Comentario principal
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                user_name = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
                date = item['snippet']['topLevelComment']['snippet']['publishedAt']
                like_count = item['snippet']['topLevelComment']['snippet'].get('likeCount', 0)
                comments.append(comment)
                user_names.append(user_name)
                dates.append(date)
                likes.append(like_count)

                # Respuestas
                replycount = item['snippet']['totalReplyCount']
                if replycount > 0 and 'replies' in item:
                    reply_list = []
                    for reply in item['replies']['comments']:
                        reply_text = reply['snippet']['textDisplay']
                        reply_list.append(reply_text)
                    replies.append(reply_list)
                else:
                    replies.append([])

            # Guardar en DataFrame y CSV
            df2 = pd.DataFrame({
                "comment": comments,
                "replies": replies,
                "user_name": user_names,
                "date": dates,
                "likes": likes
            })
            df = pd.concat([df, df2], ignore_index=True)
            df.to_csv(f"{video_id}_user_comments.csv", index=False, encoding='utf-8')
            # Limpiar listas para la siguiente página
            comments, replies, user_names, dates, likes = [], [], [], [], []
            sleep(2)
            request = youtube.commentThreads().list_next(request, response)
            print("Iterando siguiente página...")
        except Exception as e:
            print(str(e))
            print(traceback.format_exc())
            print("Esperando 10 segundos...")
            sleep(10)
            df.to_csv(f"{video_id}_user_comments.csv", index=False, encoding='utf-8')
            break
        
def adaptar_csv_para_streamlit(nombre_csv):
    df = pd.read_csv(nombre_csv)
    df['user_id'] = df['user_name']
    df['text'] = df['comment']
    df['timestamp'] = df['date']
    df['replies'] = df['replies'].apply(lambda x: len(eval(x)) if isinstance(x, str) and x.startswith('[') else 0)
    df_final = df[['user_id', 'text', 'timestamp', 'likes', 'replies']]
    df_final.to_csv("youtube_para_streamlit.csv", index=False)
    print("Archivo adaptado guardado como youtube_para_streamlit.csv")


if __name__ == "__main__":
    # Pega aquí tu API key y el ID del video
    API_KEY = "AIzaSyALRGIQsSWMoWpLKawlDDfZdZjIwMfG9YE"
    VIDEO_ID = "3H8v5hziohk"
    get_comments(API_KEY, VIDEO_ID)
    # Adaptar el CSV generado automáticamente
    nombre_csv = f"{VIDEO_ID}_user_comments.csv"
    adaptar_csv_para_streamlit(nombre_csv)


# Para adaptar el CSV al formato de la app Streamlit:
# 1. Renombra las columnas: user_name -> user_id, comment -> text, date -> timestamp
# 2. replies: cuenta la cantidad de respuestas (len), shares pon 0
# 3. El campo likes ya está incluido
# 4. Guarda el archivo con las columnas: user_id, text, timestamp, likes, replies, shares
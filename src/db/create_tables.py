from src.db.database import engine, Base

def run():
    print("Criando tabelas no banco de dados...")
    Base.metadata.create_all(bind=engine)
    print("Tabelas criadas!")

if __name__ == "__main__":
    run()

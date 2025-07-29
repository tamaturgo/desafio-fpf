from src.db.database import engine, Base

def run():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    run()

from sklearn.model_selection import train_test_split

class ModelTrainer:
    def __init__(self, model_path):
        self.model_path = model_path
        
    def train(self, processed_texts):
       train, test = train_test_split(processed_texts)
       
       # Build word vectors
       # Build model      
       # Train model
        
    def retrain(self):
        # Retrain existing model  
        
    def save_model(self):
        # Save model to path
        
    def load_model(self):
       # Load existing model  
       
    def log_performance(self):
       # Log model's accuracy
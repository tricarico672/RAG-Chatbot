-- _history table
CREATE TABLE chat_history (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users (id),
  chat_history JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
);


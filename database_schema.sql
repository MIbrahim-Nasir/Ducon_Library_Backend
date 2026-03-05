CREATE TABLE users (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  name varchar NOT NULL,
  email varchar NOT NULL UNIQUE,
  password_hash varchar,
  role varchar NOT NULL DEFAULT 'customer',
  google_id varchar UNIQUE,
  user_consent boolean NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL 
);

CREATE TABLE images (
	id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
	name varchar,
	filename varchar NOT NULL,
	url varchar NOT NULL UNIQUE,
	uploaded_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

CREATE TABLE bookmarks (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  user_id bigint REFERENCES users (id) ON DELETE CASCADE,
  image_id bigint REFERENCES images (id) ON DELETE CASCADE,
  UNIQUE (user_id, image_id)
);

CREATE TABLE generations (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  user_id bigint REFERENCES users (id) ON DELETE CASCADE,
  generation_name varchar NOT NULL,
  url varchar NOT NULL UNIQUE,
  generated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  ducon_image_id bigint REFERENCES images (id) ON DELETE SET NULL
);

CREATE INDEX idx_bookmarks_user_id ON bookmarks(user_id);
CREATE INDEX idx_generations_user_id ON generations(user_id);

CREATE INDEX idx_bookmarks_image_id ON bookmarks(image_id);
CREATE INDEX idx_generations_ducon_image_id ON generations(ducon_image_id);

